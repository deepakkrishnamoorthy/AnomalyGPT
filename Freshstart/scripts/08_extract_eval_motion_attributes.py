"""Extract EVAL-style optical-flow motion attributes for each video volume.

This is the direct optical-flow baseline. It follows the target construction
described in EVAL's motion model section:

- Yang: 12-bin normalized motion-direction histogram
- Yspeed: 12-bin average flow magnitude per direction
- Ybkg.pix: fraction of stationary pixels
- Ybkg.cls: binary background/no-motion label

These attributes are useful directly for the first anomaly baseline, and later
as self-supervised targets for a lightweight 3D CNN.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_selected_rows(path: Path, *, split: str, limit: int | None):
    selected = 0
    for row in iter_manifest(path):
        if split != "all" and row["split"] != split:
            continue
        yield row
        selected += 1
        if limit is not None and selected >= limit:
            break


def read_gray_crop(frame_path: Path, row: dict) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_path}")

    x = int(row["x"])
    y = int(row["y"])
    w = int(row["w"])
    h = int(row["h"])
    source_h, source_w = frame.shape[:2]

    pad_bottom = max(0, y + h - source_h)
    pad_right = max(0, x + w - source_w)
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            pad_bottom,
            0,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    return frame[y : y + h, x : x + w]


def load_gray_volume(row: dict, frames_root: Path) -> list[np.ndarray]:
    split_dir = SPLIT_TO_FRAME_DIR[row["split"]]
    video_dir = frames_root / split_dir / row["video_id"]
    frames = []
    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame_path = video_dir / f"{frame_idx:05d}.jpg"
        frames.append(read_gray_crop(frame_path, row))
    return frames


def make_flow_fn(method: str):
    method = method.lower()
    if method == "farneback":
        return lambda prev, curr: cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if method == "tvl1":
        if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            raise ValueError("TV-L1 requested, but cv2.optflow.DualTVL1OpticalFlow_create is not available.")
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return lambda prev, curr: tvl1.calc(prev, curr, None)

    raise ValueError(f"Unknown optical flow method: {method}")


def empty_motion_attributes(background_threshold: float) -> dict:
    out = {
        "motion_background_cls": 1,
        "motion_stationary_fraction": 1.0,
        "motion_moving_fraction": 0.0,
        "motion_mean_magnitude": 0.0,
        "motion_max_magnitude": 0.0,
        "motion_flow_method": "",
        "motion_magnitude_threshold": 0.0,
        "motion_background_threshold": background_threshold,
    }
    for idx in range(12):
        out[f"motion_angle_hist_{idx:02d}"] = 0.0
        out[f"motion_speed_{idx:02d}"] = 0.0
    return out


def compute_motion_attributes(
    frames: list[np.ndarray],
    *,
    flow_method: str,
    magnitude_threshold: float,
    background_threshold: float,
) -> dict:
    if len(frames) < 2:
        out = empty_motion_attributes(background_threshold)
        out["motion_flow_method"] = flow_method
        out["motion_magnitude_threshold"] = magnitude_threshold
        return out

    flow_fn = make_flow_fn(flow_method)
    angle_counts = np.zeros(12, dtype=np.float64)
    speed_sums = np.zeros(12, dtype=np.float64)
    speed_counts = np.zeros(12, dtype=np.float64)
    stationary_pixels = 0
    total_pixels = 0
    all_magnitude_sum = 0.0
    all_magnitude_max = 0.0

    for prev, curr in zip(frames, frames[1:]):
        flow = flow_fn(prev, curr)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        stationary = mag < magnitude_threshold
        moving = ~stationary

        stationary_pixels += int(stationary.sum())
        total_pixels += int(mag.size)
        all_magnitude_sum += float(mag.sum())
        all_magnitude_max = max(all_magnitude_max, float(mag.max()))

        bins = np.floor((ang % (2.0 * np.pi)) / (np.pi / 6.0)).astype(np.int32)
        for bin_id in range(12):
            mask = moving & (bins == bin_id)
            count = int(mask.sum())
            angle_counts[bin_id] += count
            if count:
                speed_sums[bin_id] += float(mag[mask].sum())
                speed_counts[bin_id] += count

    if total_pixels == 0:
        out = empty_motion_attributes(background_threshold)
        out["motion_flow_method"] = flow_method
        out["motion_magnitude_threshold"] = magnitude_threshold
        return out

    stationary_fraction = stationary_pixels / total_pixels
    moving_fraction = 1.0 - stationary_fraction
    angle_hist = angle_counts / total_pixels
    speed = np.divide(speed_sums, speed_counts, out=np.zeros_like(speed_sums), where=speed_counts > 0)

    out = {
        "motion_background_cls": int(stationary_fraction >= background_threshold),
        "motion_stationary_fraction": float(stationary_fraction),
        "motion_moving_fraction": float(moving_fraction),
        "motion_mean_magnitude": float(all_magnitude_sum / total_pixels),
        "motion_max_magnitude": float(all_magnitude_max),
        "motion_flow_method": flow_method,
        "motion_magnitude_threshold": float(magnitude_threshold),
        "motion_background_threshold": float(background_threshold),
    }
    for idx in range(12):
        out[f"motion_angle_hist_{idx:02d}"] = float(angle_hist[idx])
        out[f"motion_speed_{idx:02d}"] = float(speed[idx])
    return out


def build_output_row(row: dict, attrs: dict) -> dict:
    out = {
        "volume_id": row["volume_id"],
        "split": row["split"],
        "video_id": row["video_id"],
        "start_frame": row["start_frame"],
        "end_frame": row["end_frame"],
        "depth": row["depth"],
        "region_id": row["region_id"],
        "x": row["x"],
        "y": row["y"],
        "w": row["w"],
        "h": row["h"],
    }
    out.update(attrs)
    return {key: round(value, 7) if isinstance(value, float) else value for key, value in out.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-csv", type=Path, default=Path("features/avenue_eval10_motion_attributes.csv"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("manifests/avenue_eval10_motion_attributes.jsonl"))
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flow-method", choices=["farneback", "tvl1"], default="farneback")
    parser.add_argument("--magnitude-threshold", type=float, default=0.8)
    parser.add_argument("--background-threshold", type=float, default=0.98)
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out_csv.open("w", newline="", encoding="utf-8") as csv_handle, args.out_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = None
        for idx, row in enumerate(iter_selected_rows(args.manifest, split=args.split, limit=args.limit), start=1):
            frames = load_gray_volume(row, args.frames_root)
            attrs = compute_motion_attributes(
                frames,
                flow_method=args.flow_method,
                magnitude_threshold=args.magnitude_threshold,
                background_threshold=args.background_threshold,
            )
            out_row = build_output_row(row, attrs)
            if writer is None:
                writer = csv.DictWriter(csv_handle, fieldnames=list(out_row.keys()))
                writer.writeheader()

            writer.writerow(out_row)
            jsonl_handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            written += 1

            if idx % 1000 == 0:
                print(f"Processed {idx} motion volumes...")

    if written == 0:
        raise SystemExit("No manifest rows selected.")

    print(f"Wrote {written} motion rows -> {args.out_csv}")
    print(f"Wrote {written} motion rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
