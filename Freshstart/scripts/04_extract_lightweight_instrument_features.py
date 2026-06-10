"""Extract lightweight EVAL-style instrument features from 10-frame volumes.

This is a compute-conscious starter. It does not use EVAL's five pretrained
attribute networks. Instead it estimates:

- stationary/moving fraction via frame differences
- 12-bin motion direction histogram via Farneback optical flow
- 12-bin average speed rays

Object-class bars are left as zero until YOLO/ByteTrack is integrated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


OBJECT_COLUMNS = [
    "object_person",
    "object_car",
    "object_cyclist",
    "object_dog",
    "object_tree",
    "object_house",
    "object_skyscraper",
    "object_bridge",
]


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def read_volume(row: dict) -> list[np.ndarray]:
    cap = cv2.VideoCapture(row["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {row['video_path']}")
    frames = []
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["start_frame"]) - 1)
    for _ in range(int(row["depth"])):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        padded = np.pad(
            gray,
            ((0, max(0, y + h - gray.shape[0])), (0, max(0, x + w - gray.shape[1]))),
            mode="constant",
        )
        crop = padded[y : y + h, x : x + w]
        frames.append(crop)
    cap.release()
    return frames


def flow_features(frames: list[np.ndarray], motion_threshold: float) -> dict:
    if len(frames) < 2:
        return empty_features(no_motion=1)

    hist = np.zeros(12, dtype=np.float64)
    speed_sum = np.zeros(12, dtype=np.float64)
    speed_count = np.zeros(12, dtype=np.float64)
    moving_pixels = 0
    total_pixels = 0

    for prev, curr in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        moving = mag >= motion_threshold
        total_pixels += mag.size
        moving_pixels += int(moving.sum())
        bins = np.floor((ang % 360.0) / 30.0).astype(np.int32)
        for bin_id in range(12):
            mask = moving & (bins == bin_id)
            count = int(mask.sum())
            hist[bin_id] += count
            if count:
                speed_sum[bin_id] += float(mag[mask].mean())
                speed_count[bin_id] += 1

    moving_fraction = moving_pixels / total_pixels if total_pixels else 0.0
    stationary_fraction = 1.0 - moving_fraction
    hist_norm = hist / hist.sum() if hist.sum() else hist
    speed = np.divide(speed_sum, speed_count, out=np.zeros_like(speed_sum), where=speed_count > 0)
    speed_norm = speed / speed.max() if speed.max() > 0 else speed

    out = {
        "stationary_panel_fraction": float(stationary_fraction),
        "moving_panel_fraction": float(moving_fraction),
        "no_motion_indicator": int(moving_pixels == 0),
    }
    for i in range(12):
        out[f"direction_hist_{i:02d}"] = float(hist_norm[i])
        out[f"speed_ray_{i:02d}"] = float(speed[i])
        out[f"speed_ray_norm_{i:02d}"] = float(speed_norm[i])
    return out


def empty_features(no_motion: int) -> dict:
    out = {
        "stationary_panel_fraction": 1.0,
        "moving_panel_fraction": 0.0,
        "no_motion_indicator": no_motion,
    }
    for i in range(12):
        out[f"direction_hist_{i:02d}"] = 0.0
        out[f"speed_ray_{i:02d}"] = 0.0
        out[f"speed_ray_norm_{i:02d}"] = 0.0
    return out


def build_feature_row(row: dict, motion_threshold: float) -> dict:
    frames = read_volume(row)
    feat = {
        "volume_id": row["volume_id"],
        "split": row["split"],
        "video_id": row["video_id"],
        "start_frame": row["start_frame"],
        "end_frame": row["end_frame"],
        "region_id": row["region_id"],
        "x": row["x"],
        "y": row["y"],
        "w": row["w"],
        "h": row["h"],
    }
    for col in OBJECT_COLUMNS:
        feat[col] = 0.0
    feat["object_unknown_motion"] = 1.0
    feat.update(flow_features(frames, motion_threshold))
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in feat.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("features/avenue_eval10_light_instrument_features.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Use for quick validation before full extraction.")
    parser.add_argument("--motion-threshold", type=float, default=0.8)
    args = parser.parse_args()

    rows = load_rows(args.manifest, args.limit)
    features = [build_feature_row(row, args.motion_threshold) for row in rows]
    if not features:
        raise RuntimeError("No rows to write.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0].keys()))
        writer.writeheader()
        writer.writerows(features)
    print(f"Wrote {len(features)} feature rows -> {args.out}")


if __name__ == "__main__":
    main()

