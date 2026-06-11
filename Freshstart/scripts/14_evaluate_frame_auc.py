"""Evaluate Avenue frame-level anomaly scores against avenue.mat intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score


def load_gt_intervals(mat_path: Path) -> dict[str, list[tuple[int, int]]]:
    mat = sio.loadmat(mat_path, squeeze_me=False)
    if "gt" not in mat:
        raise RuntimeError(f"No 'gt' key found in {mat_path}")
    gt = mat["gt"]
    intervals = {}
    for idx in range(gt.shape[1]):
        arr = np.asarray(gt[0, idx])
        video_id = f"{idx + 1:02d}"
        if arr.size == 0:
            intervals[video_id] = []
            continue
        if arr.shape[0] != 2:
            raise RuntimeError(f"Unexpected GT shape for test video {video_id}: {arr.shape}")
        starts = arr[0].astype(int).tolist()
        ends = arr[1].astype(int).tolist()
        intervals[video_id] = list(zip(starts, ends))
    return intervals


def labels_for_video(frame_count: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.uint8)
    for start, end in intervals:
        start = max(1, int(start))
        end = min(frame_count, int(end))
        if end >= start:
            labels[start - 1 : end] = 1
    return labels


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0:
        return None
    return float(average_precision_score(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-scores", type=Path, default=Path("outputs/avenue_eval10_frame_scores.csv"))
    parser.add_argument("--gt", type=Path, default=Path("Avenue Dataset/avenue.mat"))
    parser.add_argument("--out-summary", type=Path, default=Path("outputs/avenue_eval10_frame_auc_summary.json"))
    parser.add_argument("--out-per-video", type=Path, default=Path("outputs/avenue_eval10_frame_auc_per_video.csv"))
    args = parser.parse_args()

    scores = pd.read_csv(args.frame_scores)
    if scores.empty:
        raise RuntimeError(f"No frame scores found in {args.frame_scores}")
    scores["video_id"] = scores["video_id"].astype(str).str.zfill(2)

    gt = load_gt_intervals(args.gt)

    all_labels = []
    all_scores = []
    per_video = []
    for video_id, group in scores.groupby("video_id"):
        group = group.sort_values("frame")
        frame_scores = group["frame_score"].to_numpy(dtype=np.float32)
        labels = labels_for_video(len(group), gt.get(video_id, []))
        auc = safe_auc(labels, frame_scores)
        ap = safe_ap(labels, frame_scores)
        per_video.append(
            {
                "video_id": video_id,
                "frames": int(len(group)),
                "anomaly_frames": int(labels.sum()),
                "normal_frames": int((labels == 0).sum()),
                "roc_auc": auc,
                "average_precision": ap,
                "score_min": float(frame_scores.min()),
                "score_mean": float(frame_scores.mean()),
                "score_max": float(frame_scores.max()),
            }
        )
        all_labels.append(labels)
        all_scores.append(frame_scores)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)
    summary = {
        "frames": int(len(y_true)),
        "anomaly_frames": int(y_true.sum()),
        "normal_frames": int((y_true == 0).sum()),
        "global_roc_auc": safe_auc(y_true, y_score),
        "global_average_precision": safe_ap(y_true, y_score),
        "videos": len(per_video),
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with args.out_per_video.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        writer.writerows(per_video)

    print(f"Wrote summary -> {args.out_summary}")
    print(f"Wrote per-video metrics -> {args.out_per_video}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
