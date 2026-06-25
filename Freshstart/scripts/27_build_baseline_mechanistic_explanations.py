"""Build three-layer explanations for the stride-10 exemplar baseline.

Layers:
1. Exemplar-level: test volume vs nearest normal exemplar.
2. Attribute-level: appearance/direction/speed/background contribution.
3. Backbone-level: ResNet18 Grad-CAM for the appearance component.

This script is intentionally honest about the model: the anomaly scorer is not
a neural network. Grad-CAM explains the pretrained ResNet18 appearance backbone,
not the nearest-exemplar anomaly detector itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms


SPLIT_TO_FRAME_DIR = {"training": "Train", "testing": "Test"}
APP_PREFIX = "appearance_resnet18_emb_"
ANGLE_COLS = [f"motion_angle_hist_{idx:02d}" for idx in range(12)]
SPEED_COLS = [f"motion_speed_{idx:02d}" for idx in range(12)]
MOTION_SCALAR_COLS = [
    "motion_background_cls",
    "motion_stationary_fraction",
    "motion_moving_fraction",
    "motion_mean_magnitude",
    "motion_max_magnitude",
]
META_COLS = ["volume_id", "split", "video_id", "start_frame", "end_frame", "region_id", "x", "y", "w", "h"]


def as_video_id(value) -> str:
    return f"{int(value):02d}"


def frame_path(frames_root: Path, row: pd.Series, frame_idx: int) -> Path:
    return frames_root / SPLIT_TO_FRAME_DIR[str(row["split"])] / as_video_id(row["video_id"]) / f"{frame_idx:05d}.jpg"


def read_frame(frames_root: Path, row: pd.Series, frame_idx: int) -> np.ndarray:
    path = frame_path(frames_root, row, frame_idx)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame {path}")
    return frame


def crop_region(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pad_bottom = max(0, y + h - frame.shape[0])
    pad_right = max(0, x + w - frame.shape[1])
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame[y : y + h, x : x + w]


def draw_region(frame: np.ndarray, row: pd.Series, label: str, color=(255, 255, 255)) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, label, (x + 4, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def contact_sheet(frames_root: Path, row: pd.Series, out_path: Path, *, full_frame: bool) -> str:
    tiles = []
    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame = read_frame(frames_root, row, frame_idx)
        tile = draw_region(frame, row, f"f{frame_idx}") if full_frame else crop_region(frame, row)
        tile = cv2.resize(tile, (160, 90), interpolation=cv2.INTER_AREA) if full_frame else cv2.resize(tile, (96, 96))
        cv2.putText(tile, str(frame_idx), (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path)


def side_by_side(test_sheet: str, exemplar_sheet: str, out_path: Path) -> str:
    left = cv2.imread(test_sheet)
    right = cv2.imread(exemplar_sheet)
    if left is None or right is None:
        raise RuntimeError("Could not read contact sheet for side-by-side image.")
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.resize(left, (left.shape[1], h), interpolation=cv2.INTER_AREA)
    if right.shape[0] != h:
        right = cv2.resize(right, (right.shape[1], h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 18, 3), 245, dtype=np.uint8)
    canvas = np.concatenate([left, gap, right], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path)


def draw_bars(panel: np.ndarray, title: str, labels: list[str], values: list[float], x: int, y: int, w: int, h: int, color) -> None:
    cv2.putText(panel, title, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 32, 40), 1, cv2.LINE_AA)
    max_value = max(max(abs(v) for v in values), 1e-6)
    gap = 4
    bar_w = max(4, (w - gap * (len(values) - 1)) // len(values))
    zero_y = y + h // 2
    cv2.line(panel, (x, zero_y), (x + w, zero_y), (200, 200, 200), 1)
    for idx, value in enumerate(values):
        bx = x + idx * (bar_w + gap)
        bh = int(abs(value) / max_value * (h // 2 - 4))
        if value >= 0:
            cv2.rectangle(panel, (bx, zero_y - bh), (bx + bar_w, zero_y), color, -1)
        else:
            cv2.rectangle(panel, (bx, zero_y), (bx + bar_w, zero_y + bh), (90, 130, 210), -1)
        cv2.putText(panel, labels[idx], (bx, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 85, 90), 1, cv2.LINE_AA)


def attribute_panel(case: dict, test_feat: pd.Series, ex_feat: pd.Series, app_cols: list[str], out_path: Path) -> tuple[str, dict]:
    distances = {
        "appearance": float(case["distance_app"]),
        "direction": float(case["distance_ang"]),
        "speed": float(case["distance_speed"]),
        "background": float(case["distance_bkg"]),
    }
    total = sum(max(0.0, v) for v in distances.values()) or 1.0
    shares = {k: v / total for k, v in distances.items()}

    test_angle = np.array([float(test_feat[c]) for c in ANGLE_COLS], dtype=np.float32)
    ex_angle = np.array([float(ex_feat[c]) for c in ANGLE_COLS], dtype=np.float32)
    test_speed = np.array([float(test_feat[c]) for c in SPEED_COLS], dtype=np.float32)
    ex_speed = np.array([float(ex_feat[c]) for c in SPEED_COLS], dtype=np.float32)
    app_delta = test_feat[app_cols].to_numpy(dtype=np.float32) - ex_feat[app_cols].to_numpy(dtype=np.float32)
    top_app = np.argsort(np.abs(app_delta))[::-1][:10]
    top_app_dims = [{"dim": int(idx), "delta": float(app_delta[idx])} for idx in top_app]

    panel = np.full((720, 1120, 3), 255, dtype=np.uint8)
    cv2.putText(panel, f"{case['case_id']} score {case['anomaly_score']:.3f}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 32, 40), 2, cv2.LINE_AA)
    cv2.putText(panel, f"region {case['region_id']} frames {case['start_frame']}-{case['end_frame']}", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 85, 90), 1, cv2.LINE_AA)

    labels = list(distances.keys())
    values = [distances[k] for k in labels]
    max_dist = max(values) or 1.0
    for idx, label in enumerate(labels):
        y = 130 + idx * 50
        width = int(330 * values[idx] / max_dist)
        cv2.putText(panel, f"{label}: {values[idx]:.3f} ({shares[label]*100:.1f}%)", (40, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 32, 40), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (250, y), (250 + width, y + 24), (35, 105, 170), -1)
        cv2.rectangle(panel, (250, y), (580, y + 24), (210, 216, 222), 1)

    draw_bars(panel, "Direction histogram delta (test - exemplar)", [str(i) for i in range(12)], (test_angle - ex_angle).tolist(), 650, 130, 360, 140, (205, 80, 40))
    draw_bars(panel, "Speed vector delta (test - exemplar)", [str(i) for i in range(12)], (test_speed - ex_speed).tolist(), 650, 365, 360, 140, (205, 80, 40))

    lines = [
        f"Test moving fraction: {float(test_feat['motion_moving_fraction']):.3f}",
        f"Exemplar moving fraction: {float(ex_feat['motion_moving_fraction']):.3f}",
        f"Test stationary fraction: {float(test_feat['motion_stationary_fraction']):.3f}",
        f"Exemplar stationary fraction: {float(ex_feat['motion_stationary_fraction']):.3f}",
        "Top appearance embedding deltas: " + ", ".join([f"{d['dim']}:{d['delta']:.2f}" for d in top_app_dims[:5]]),
    ]
    for idx, line in enumerate(lines):
        cv2.putText(panel, line, (40, 390 + idx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 52, 60), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path), {
        "distances": distances,
        "distance_shares": shares,
        "direction_delta": (test_angle - ex_angle).astype(float).tolist(),
        "speed_delta": (test_speed - ex_speed).astype(float).tolist(),
        "moving_fraction_delta": float(test_feat["motion_moving_fraction"] - ex_feat["motion_moving_fraction"]),
        "stationary_fraction_delta": float(test_feat["motion_stationary_fraction"] - ex_feat["motion_stationary_fraction"]),
        "top_appearance_embedding_deltas": top_app_dims,
    }


class GradCamResNet18:
    def __init__(self, device: torch.device):
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.device = device
        self.activations = None
        self.gradients = None
        self.model.layer4.register_forward_hook(self._save_activations)
        self.model.layer4.register_full_backward_hook(self._save_gradients)
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _save_activations(self, _module, _inp, out):
        self.activations = out

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def overlay_for_embedding_dim(self, crop_bgr: np.ndarray, dim: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self.preprocess(crop_rgb).unsqueeze(0).to(self.device)
        self.model.zero_grad(set_to_none=True)
        emb = self.model(x)
        target = emb[0, dim] * float(sign)
        target.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cv2.resize(cam, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(crop_bgr, 0.58, heat, 0.42, 0)
        return overlay, cam


def backbone_gradcam(
    gradcam: GradCamResNet18,
    frames_root: Path,
    test_feat: pd.Series,
    ex_feat: pd.Series,
    app_delta: list[dict],
    out_dir: Path,
    case_id: str,
) -> dict:
    selected = app_delta[0]
    dim = int(selected["dim"])
    sign = 1.0 if float(selected["delta"]) >= 0 else -1.0
    test_frame_idx = int(round((int(test_feat["start_frame"]) + int(test_feat["end_frame"])) / 2))
    ex_frame_idx = int(round((int(ex_feat["start_frame"]) + int(ex_feat["end_frame"])) / 2))
    test_crop = crop_region(read_frame(frames_root, test_feat, test_frame_idx), test_feat)
    ex_crop = crop_region(read_frame(frames_root, ex_feat, ex_frame_idx), ex_feat)
    test_overlay, _ = gradcam.overlay_for_embedding_dim(test_crop, dim, sign)
    ex_overlay, _ = gradcam.overlay_for_embedding_dim(ex_crop, dim, sign)

    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / f"{case_id}_test_resnet_gradcam.jpg"
    ex_path = out_dir / f"{case_id}_exemplar_resnet_gradcam.jpg"
    side_path = out_dir / f"{case_id}_resnet_gradcam_side_by_side.jpg"
    cv2.imwrite(str(test_path), test_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(ex_path), ex_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    gap = np.full((test_overlay.shape[0], 16, 3), 245, dtype=np.uint8)
    combined = np.concatenate([test_overlay, gap, ex_overlay], axis=1)
    cv2.imwrite(str(side_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return {
        "target_embedding_dim": dim,
        "target_sign": sign,
        "test_frame": test_frame_idx,
        "exemplar_frame": ex_frame_idx,
        "test_gradcam": str(test_path),
        "exemplar_gradcam": str(ex_path),
        "side_by_side_gradcam": str(side_path),
        "scope": "ResNet18 appearance-backbone interpretation, not anomaly-detector circuit analysis.",
    }


def select_cases(scores: pd.DataFrame, top_k: int, per_video_k: int | None, dedupe_window: int) -> pd.DataFrame:
    def select_group(group: pd.DataFrame, k: int) -> list[pd.Series]:
        group = group.sort_values("anomaly_score", ascending=False)
        chosen = []
        seen: list[tuple[int, int]] = []
        for _, row in group.iterrows():
            key = (int(row["region_id"]), int(row["start_frame"]))
            if any(key[0] == r and abs(key[1] - s) <= dedupe_window for r, s in seen):
                continue
            chosen.append(row)
            seen.append(key)
            if len(chosen) >= k:
                break
        return chosen

    if per_video_k is not None:
        rows = []
        for _, group in scores.groupby(scores["video_id"].astype(int), sort=True):
            rows.extend(select_group(group, per_video_k))
        return pd.DataFrame(rows)
    return pd.DataFrame(select_group(scores, top_k))


def load_needed_features(features_path: Path, volume_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(features_path, nrows=0)
    app_cols = sorted([c for c in header.columns if c.startswith(APP_PREFIX)])
    usecols = META_COLS + MOTION_SCALAR_COLS + ANGLE_COLS + SPEED_COLS + app_cols
    features = pd.read_csv(features_path, usecols=usecols)
    features["volume_id"] = features["volume_id"].astype(str)
    features = features[features["volume_id"].isin(volume_ids)].set_index("volume_id")
    return features, app_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/baseline_mechanistic_explainability"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--per-video-k", type=int, default=None)
    parser.add_argument("--dedupe-window", type=int, default=30)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    selected = select_cases(scores, args.top_k, args.per_video_k, args.dedupe_window)
    if selected.empty:
        raise SystemExit("No cases selected.")
    needed_ids = set(selected["volume_id"].astype(str)) | set(selected["nearest_exemplar_volume_id"].astype(str))
    features, app_cols = load_needed_features(args.features, needed_ids)
    gradcam = None if args.skip_gradcam else GradCamResNet18(torch.device(args.device))

    exemplar_dir = args.out_dir / "exemplar_level"
    attribute_dir = args.out_dir / "attribute_level"
    backbone_dir = args.out_dir / "backbone_level"
    reports_dir = args.out_dir / "reports"
    cases = []

    for idx, score_row in enumerate(selected.itertuples(index=False), start=1):
        row = score_row._asdict()
        volume_id = str(row["volume_id"])
        exemplar_id = str(row["nearest_exemplar_volume_id"])
        test_feat = features.loc[volume_id]
        ex_feat = features.loc[exemplar_id]
        case_id = f"case_{idx:03d}"
        case = {
            "case_id": case_id,
            "volume_id": volume_id,
            "nearest_exemplar_volume_id": exemplar_id,
            "video_id": as_video_id(row["video_id"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "region_id": int(row["region_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "w": int(row["w"]),
            "h": int(row["h"]),
            "anomaly_score": float(row["anomaly_score"]),
            "distance_app": float(row["distance_app"]),
            "distance_ang": float(row["distance_ang"]),
            "distance_speed": float(row["distance_speed"]),
            "distance_bkg": float(row["distance_bkg"]),
        }

        test_sheet = contact_sheet(args.frames_root, test_feat, exemplar_dir / f"{case_id}_test_sheet.jpg", full_frame=True)
        ex_sheet = contact_sheet(args.frames_root, ex_feat, exemplar_dir / f"{case_id}_nearest_normal_sheet.jpg", full_frame=True)
        comparison = side_by_side(test_sheet, ex_sheet, exemplar_dir / f"{case_id}_test_vs_nearest_normal.jpg")
        attr_panel, attr_json = attribute_panel(case, test_feat, ex_feat, app_cols, attribute_dir / f"{case_id}_attribute_panel.jpg")

        case["exemplar_level"] = {
            "test_sheet": test_sheet,
            "nearest_normal_sheet": ex_sheet,
            "side_by_side": comparison,
            "interpretation": "Nearest normal exemplar comparison within the same spatial region.",
        }
        case["attribute_level"] = {
            "attribute_panel": attr_panel,
            **attr_json,
        }
        if gradcam is not None:
            case["backbone_level"] = backbone_gradcam(
                gradcam,
                args.frames_root,
                test_feat,
                ex_feat,
                attr_json["top_appearance_embedding_deltas"],
                backbone_dir,
                case_id,
            )
        else:
            case["backbone_level"] = {"skipped": True}
        cases.append(case)
        print(f"Built {case_id}: score={case['anomaly_score']:.3f} region={case['region_id']}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "baseline_mechanistic_explanations.json"
    report_path.write_text(json.dumps(cases, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
