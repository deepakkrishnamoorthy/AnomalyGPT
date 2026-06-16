"""Profile runtime and algorithmic complexity for the Freshstart VAD baseline.

This script answers three practical questions:

1. What are the actual stride-10 dataset/model sizes?
2. Which parts use GPU vs CPU in our implementation?
3. How expensive are appearance extraction, Farneback motion attributes, and
   exemplar inference when measured on small samples and extrapolated?

The default run is intentionally light. It reads existing artifacts and times
small samples instead of recomputing the full pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def now() -> float:
    return time.perf_counter()


def seconds_to_text(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_manifest(path: Path) -> dict[str, Any]:
    total = 0
    by_split: dict[str, int] = {}
    by_video: dict[str, int] = {}
    region_ids = set()
    starts_by_video: dict[tuple[str, str], set[int]] = {}
    depth_values = set()
    region_size_values = set()

    for row in iter_jsonl(path):
        total += 1
        split = str(row["split"])
        video_id = f"{int(row['video_id']):02d}"
        by_split[split] = by_split.get(split, 0) + 1
        by_video[f"{split}_{video_id}"] = by_video.get(f"{split}_{video_id}", 0) + 1
        region_ids.add(int(row["region_id"]))
        starts_by_video.setdefault((split, video_id), set()).add(int(row["start_frame"]))
        depth_values.add(int(row["depth"]))
        region_size_values.add((int(row["w"]), int(row["h"])))

    temporal_windows = {f"{split}_{video}": len(starts) for (split, video), starts in starts_by_video.items()}
    return {
        "manifest_rows": total,
        "rows_by_split": by_split,
        "rows_by_video": by_video,
        "region_count": len(region_ids),
        "region_ids_minmax": [min(region_ids), max(region_ids)] if region_ids else None,
        "temporal_windows_by_video": temporal_windows,
        "temporal_windows_total": int(sum(temporal_windows.values())),
        "depth_values": sorted(depth_values),
        "region_size_values": sorted([list(v) for v in region_size_values]),
    }


def load_model_summary(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    region_exemplars = {int(k): len(v["volume_ids"]) for k, v in model["regions"].items()}
    component_columns = model["component_columns"]
    dims = {name: len(cols) for name, cols in component_columns.items()}
    score_dim = dims["app"] + dims["ang"] + dims["speed"] + dims["bkg"]
    return {
        "model": model,
        "region_exemplars": region_exemplars,
        "total_exemplars": int(sum(region_exemplars.values())),
        "min_exemplars_per_region": int(min(region_exemplars.values())),
        "max_exemplars_per_region": int(max(region_exemplars.values())),
        "mean_exemplars_per_region": float(statistics.mean(region_exemplars.values())),
        "component_dims": dims,
        "distance_feature_dim": int(score_dim),
        "normalizers": model["normalizers"],
        "background_mismatch_penalty": float(model.get("background_mismatch_penalty", 3.0)),
    }


def gpu_report() -> dict[str, Any]:
    report = {
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "note": "Appearance extraction can use CUDA through PyTorch. Farneback motion and exemplar scoring are CPU/Numpy/OpenCV in this repo.",
    }
    try:
        import torch

        report["torch_available"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
        if report["cuda_available"]:
            report["devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["torch_error"] = str(exc)
    return report


def row_components(row: pd.Series, comps: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        "app": row[comps["app"]].to_numpy(dtype=np.float32),
        "ang": row[comps["ang"]].to_numpy(dtype=np.float32),
        "speed": row[comps["speed"]].to_numpy(dtype=np.float32),
        "bkg": row[comps["bkg"]].to_numpy(dtype=np.float32),
        "cls": row[comps["cls"]].to_numpy(dtype=np.float32),
    }


def component_distances(
    candidate: dict[str, np.ndarray],
    exemplars: dict,
    normalizers: dict[str, float],
    *,
    mismatch_penalty: float,
) -> np.ndarray:
    d_app = np.linalg.norm(exemplars["app"] - candidate["app"], axis=1) / normalizers["app"]
    d_ang = np.linalg.norm(exemplars["ang"] - candidate["ang"], axis=1) / normalizers["ang"]
    d_speed = np.linalg.norm(exemplars["speed"] - candidate["speed"], axis=1) / normalizers["speed"]
    d_bkg = np.linalg.norm(exemplars["bkg"] - candidate["bkg"], axis=1) / normalizers["bkg"]
    candidate_background = int(candidate["cls"][0]) == 1
    exemplar_background = exemplars["cls"].reshape(-1).astype(np.int32) == 1

    both_background = candidate_background & exemplar_background
    both_motion = (not candidate_background) & (~exemplar_background)
    mismatch = ~(both_background | both_motion)

    total = np.zeros_like(d_app)
    total[both_background] = d_app[both_background]
    total[both_motion] = d_app[both_motion] + d_ang[both_motion] + d_speed[both_motion] + d_bkg[both_motion]
    total[mismatch] = d_app[mismatch] + mismatch_penalty
    return total


def benchmark_scoring(features_path: Path, model_info: dict[str, Any], sample_rows: int) -> dict[str, Any]:
    model = model_info["model"]
    comps = model["component_columns"]
    usecols = [
        "volume_id",
        "split",
        "video_id",
        "start_frame",
        "end_frame",
        "region_id",
        *comps["app"],
        *comps["ang"],
        *comps["speed"],
        *comps["bkg"],
        *comps["cls"],
    ]
    df = pd.read_csv(features_path, usecols=usecols)
    test = df[df["split"] == "testing"].head(sample_rows).copy()
    if test.empty:
        return {"available": False, "reason": "No testing rows found."}

    normalizers = model["normalizers"]
    mismatch_penalty = float(model.get("background_mismatch_penalty", 3.0))
    comparisons = 0
    t0 = now()
    best_scores = []
    for _, row in test.iterrows():
        region_id = int(row["region_id"])
        exemplars = model["regions"][region_id]
        candidate = row_components(row, comps)
        distances = component_distances(candidate, exemplars, normalizers, mismatch_penalty=mismatch_penalty)
        best_scores.append(float(distances.min()))
        comparisons += int(len(distances))
    elapsed = now() - t0

    rows_per_second = len(test) / elapsed
    comparisons_per_second = comparisons / elapsed
    total_test_rows = int((df["split"] == "testing").sum())
    estimated_full_test_seconds = total_test_rows / rows_per_second
    return {
        "available": True,
        "sample_rows": int(len(test)),
        "elapsed_seconds": elapsed,
        "seconds_per_volume": elapsed / len(test),
        "volumes_per_second": rows_per_second,
        "exemplar_comparisons": comparisons,
        "exemplar_comparisons_per_second": comparisons_per_second,
        "score_mean_sample": float(statistics.mean(best_scores)),
        "estimated_full_test_rows": total_test_rows,
        "estimated_full_test_seconds": estimated_full_test_seconds,
    }


def read_gray_crop(frame_path: Path, row: dict) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_path}")
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pad_bottom = max(0, y + h - frame.shape[0])
    pad_right = max(0, x + w - frame.shape[1])
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
    return frame[y : y + h, x : x + w]


def load_gray_volume(row: dict, frames_root: Path) -> list[np.ndarray]:
    video_dir = frames_root / SPLIT_TO_FRAME_DIR[row["split"]] / f"{int(row['video_id']):02d}"
    return [read_gray_crop(video_dir / f"{idx:05d}.jpg", row) for idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1)]


def compute_farneback_volume(frames: list[np.ndarray]) -> None:
    for prev, curr in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        _ = (mag > 0.8).sum() + (ang >= 0).sum()


def sample_manifest_rows(manifest_path: Path, split: str, sample_rows: int) -> list[dict]:
    rows = []
    for row in iter_jsonl(manifest_path):
        if split != "all" and row["split"] != split:
            continue
        rows.append(row)
        if len(rows) >= sample_rows:
            break
    return rows


def benchmark_motion(manifest_path: Path, frames_root: Path, sample_rows: int) -> dict[str, Any]:
    rows = sample_manifest_rows(manifest_path, "testing", sample_rows)
    if not rows:
        return {"available": False, "reason": "No testing manifest rows found."}

    t0 = now()
    pixels = 0
    flow_pairs = 0
    for row in rows:
        frames = load_gray_volume(row, frames_root)
        compute_farneback_volume(frames)
        pixels += int(row["w"]) * int(row["h"]) * max(0, len(frames) - 1)
        flow_pairs += max(0, len(frames) - 1)
    elapsed = now() - t0
    return {
        "available": True,
        "sample_rows": len(rows),
        "elapsed_seconds": elapsed,
        "seconds_per_volume": elapsed / len(rows),
        "volumes_per_second": len(rows) / elapsed,
        "flow_pairs": flow_pairs,
        "pixels_processed": pixels,
    }


def read_rgb_crop(frame_path: Path, row: dict) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_path}")
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pad_bottom = max(0, y + h - frame.shape[0])
    pad_right = max(0, x + w - frame.shape[1])
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = frame[y : y + h, x : x + w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def benchmark_appearance(manifest_path: Path, frames_root: Path, sample_rows: int, device_name: str) -> dict[str, Any]:
    try:
        import torch
        from torchvision import models, transforms
    except Exception as exc:
        return {"available": False, "reason": f"PyTorch/torchvision unavailable: {exc}"}

    rows = sample_manifest_rows(manifest_path, "testing", sample_rows)
    if not rows:
        return {"available": False, "reason": "No testing manifest rows found."}

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frame_count = 0
    t0 = now()
    with torch.no_grad():
        for row in rows:
            video_dir = frames_root / SPLIT_TO_FRAME_DIR[row["split"]] / f"{int(row['video_id']):02d}"
            tensors = []
            for idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
                crop = read_rgb_crop(video_dir / f"{idx:05d}.jpg", row)
                tensors.append(transform(crop))
            batch = torch.stack(tensors, dim=0).to(device)
            _ = net(batch).mean(dim=0)
            frame_count += len(tensors)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = now() - t0
    return {
        "available": True,
        "device": str(device),
        "sample_rows": len(rows),
        "frames_forwarded": frame_count,
        "elapsed_seconds": elapsed,
        "seconds_per_volume": elapsed / len(rows),
        "volumes_per_second": len(rows) / elapsed,
        "frames_per_second": frame_count / elapsed,
    }


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def add_full_test_estimates(benchmarks: dict[str, Any], total_test_rows: int, test_frame_count: int) -> None:
    total_seconds = 0.0
    has_full_pipeline = True
    for name in ["appearance", "motion", "scoring"]:
        result = benchmarks.get(name, {})
        if not result.get("available"):
            has_full_pipeline = False
            continue
        result.setdefault("estimated_full_test_rows", total_test_rows)
        result["estimated_full_test_seconds"] = float(result["seconds_per_volume"]) * total_test_rows
        result["estimated_frame_equivalent_fps"] = (
            test_frame_count / result["estimated_full_test_seconds"] if result["estimated_full_test_seconds"] > 0 else None
        )
        total_seconds += result["estimated_full_test_seconds"]

    if has_full_pipeline:
        benchmarks["end_to_end_online_estimate"] = {
            "available": True,
            "estimated_full_test_seconds": total_seconds,
            "estimated_frame_equivalent_fps": test_frame_count / total_seconds if total_seconds > 0 else None,
            "note": "Appearance + Farneback motion + exemplar scoring using current unoptimized per-volume scripts.",
        }


def build_complexity_notes(manifest: dict[str, Any], model: dict[str, Any]) -> list[str]:
    regions = manifest["region_count"]
    depth = manifest["depth_values"][0] if len(manifest["depth_values"]) == 1 else "L"
    dim = model["distance_feature_dim"]
    mean_e = model["mean_exemplars_per_region"]
    max_e = model["max_exemplars_per_region"]
    return [
        f"Let V be number of video volumes, R spatial regions ({regions}), L clip depth ({depth}), P=h*w pixels per region, d distance feature dimension ({dim}), and E_r exemplars in a region.",
        "Manifest generation: O(number_of_videos + temporal_windows * R).",
        "Appearance feature extraction: O(V * L * ResNet18_forward). In this repo it is the only deep model path and can run on GPU through PyTorch/CUDA.",
        "Motion feature extraction: O(V * (L-1) * P * I_flow), where I_flow is the Farneback optical-flow work per pixel. In this repo it runs on CPU through OpenCV.",
        "Model building / greedy exemplar selection: O(sum_r N_r * E_r * d), worst case O(sum_r N_r^2 * d) if every normal volume becomes an exemplar.",
        f"Exemplar inference/scoring: O(sum_test_volumes E_region(volume) * d). With this model mean E_r is {mean_e:.1f}, max E_r is {max_e}.",
        "Frame/spatial projection: O(V * L) for frame max scores, plus a small constant for writing each region score to the compact spatial grid.",
        "Memory for exemplar model: O(sum_r E_r * d). Memory for full CSV features is O(V * d) on disk; the current scripts load this CSV into RAM for scoring.",
    ]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Freshstart VAD Complexity Profile",
        "",
        "## What This Baseline Does",
        "",
        "This is a lightweight EVAL-inspired pipeline. We use 10-frame spatio-temporal volumes, ResNet18 ImageNet features for appearance, Farneback optical-flow attributes for motion, and region-specific normal exemplar matching for anomaly scoring.",
        "",
        "Unlike EVAL, we do not train separate deep networks for appearance, direction, speed, or background. Appearance uses one pretrained backbone; motion attributes are computed directly from optical flow.",
        "",
        "## Device Use",
        "",
        f"- Torch available: `{report['gpu']['torch_available']}`",
        f"- CUDA available: `{report['gpu']['cuda_available']}`",
        f"- CUDA devices: `{report['gpu']['devices']}`",
        "- GPU path: ResNet18 appearance extraction only, when `--device cuda` is used.",
        "- CPU path: frame I/O/cropping, Farneback motion extraction, exemplar training/scoring, frame projection, evaluation.",
        "",
        "## Actual Stride-10 Sizes",
        "",
        f"- Manifest rows / volumes: `{report['manifest']['manifest_rows']}`",
        f"- Rows by split: `{report['manifest']['rows_by_split']}`",
        f"- Region count: `{report['manifest']['region_count']}`",
        f"- Temporal windows total: `{report['manifest']['temporal_windows_total']}`",
        f"- Total exemplars: `{report['model']['total_exemplars']}`",
        f"- Mean exemplars per region: `{report['model']['mean_exemplars_per_region']:.2f}`",
        f"- Max exemplars per region: `{report['model']['max_exemplars_per_region']}`",
        "",
        "## Benchmarks",
        "",
    ]
    for name in ["appearance", "motion", "scoring"]:
        result = report["benchmarks"].get(name, {})
        lines.append(f"### {name.title()}")
        if not result.get("available", False):
            lines.append(f"- Not available: {result.get('reason', 'unknown')}")
        else:
            lines.append(f"- Sample rows: `{result.get('sample_rows')}`")
            lines.append(f"- Elapsed: `{seconds_to_text(result.get('elapsed_seconds', 0.0))}`")
            lines.append(f"- Seconds per volume: `{result.get('seconds_per_volume', 0.0):.6f}`")
            lines.append(f"- Volumes/sec: `{result.get('volumes_per_second', 0.0):.2f}`")
            if "estimated_full_test_seconds" in result:
                lines.append(f"- Estimated full test time: `{seconds_to_text(result['estimated_full_test_seconds'])}`")
            if "estimated_frame_equivalent_fps" in result and result["estimated_frame_equivalent_fps"] is not None:
                lines.append(f"- Estimated frame-equivalent FPS: `{result['estimated_frame_equivalent_fps']:.2f}`")
            if "device" in result:
                lines.append(f"- Device: `{result['device']}`")
        lines.append("")

    online = report["benchmarks"].get("end_to_end_online_estimate", {})
    if online.get("available"):
        lines.extend(
            [
                "### End-To-End Online Estimate",
                f"- Estimated full test time: `{seconds_to_text(online['estimated_full_test_seconds'])}`",
                f"- Estimated frame-equivalent FPS: `{online['estimated_frame_equivalent_fps']:.2f}`",
                f"- Note: {online['note']}",
                "",
            ]
        )

    lines.extend(["## Big-O Complexity", ""])
    lines.extend([f"- {note}" for note in report["complexity_notes"]])
    lines.extend(
        [
            "",
            "## Comparison Caveat",
            "",
            "This profile is useful for comparing our own stride-10 and stride-1 ablations. It is not a strict SOTA runtime comparison because papers may use different frame rates, region sizes, temporal strides, hardware, optimized batching, or learned feature extractors.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/avenue_eval10_region_exemplars.pkl"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/complexity_profile"))
    parser.add_argument("--scoring-sample", type=int, default=2000)
    parser.add_argument("--motion-sample", type=int, default=200)
    parser.add_argument("--appearance-sample", type=int, default=50)
    parser.add_argument("--skip-appearance", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda for appearance benchmark.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_info = count_manifest(args.manifest)
    model_info = load_model_summary(args.model)
    gpu_info = gpu_report()
    artifact_rows = {
        "appearance_feature_rows": csv_row_count(Path("features/avenue_eval10_appearance_resnet18.csv")),
        "motion_feature_rows": csv_row_count(Path("features/avenue_eval10_motion_attributes.csv")),
        "model_feature_rows": csv_row_count(args.features),
        "score_rows": csv_row_count(Path("outputs/avenue_eval10_exemplar_scores.csv")),
        "frame_score_rows": csv_row_count(Path("outputs/avenue_eval10_frame_scores.csv")),
    }

    benchmarks: dict[str, Any] = {
        "scoring": benchmark_scoring(args.features, model_info, args.scoring_sample),
    }
    if args.skip_motion:
        benchmarks["motion"] = {"available": False, "reason": "Skipped by user."}
    else:
        benchmarks["motion"] = benchmark_motion(args.manifest, args.frames_root, args.motion_sample)

    if args.skip_appearance:
        benchmarks["appearance"] = {"available": False, "reason": "Skipped by user."}
    else:
        benchmarks["appearance"] = benchmark_appearance(args.manifest, args.frames_root, args.appearance_sample, args.device)

    add_full_test_estimates(
        benchmarks,
        int(manifest_info["rows_by_split"].get("testing", 0)),
        int(artifact_rows["frame_score_rows"]),
    )

    report = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
        },
        "gpu": gpu_info,
        "manifest": manifest_info,
        "artifacts": artifact_rows,
        "model": {k: v for k, v in model_info.items() if k != "model"},
        "benchmarks": benchmarks,
        "complexity_notes": build_complexity_notes(manifest_info, model_info),
    }

    json_path = args.out_dir / "complexity_profile.json"
    md_path = args.out_dir / "complexity_profile.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, report)

    print(f"Wrote JSON -> {json_path}")
    print(f"Wrote report -> {md_path}")
    print()
    print("Key runtime summary:")
    for name in ["appearance", "motion", "scoring"]:
        result = benchmarks.get(name, {})
        if result.get("available"):
            print(
                f"- {name}: {result['volumes_per_second']:.2f} volumes/sec "
                f"({result['seconds_per_volume']:.6f} sec/volume)"
            )
        else:
            print(f"- {name}: unavailable ({result.get('reason', 'unknown')})")
    print()
    print("GPU summary:")
    print(f"- CUDA available: {gpu_info['cuda_available']}")
    if gpu_info["devices"]:
        for device in gpu_info["devices"]:
            print(f"- GPU {device['index']}: {device['name']} ({device['memory_gb']} GB)")


if __name__ == "__main__":
    main()
