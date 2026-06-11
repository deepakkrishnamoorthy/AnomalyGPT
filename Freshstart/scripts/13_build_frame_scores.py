"""Convert region/volume anomaly scores into frame-level scores.

For each test video, every scored 10-frame volume contributes its anomaly score
to all frames covered by that volume. If multiple regions/windows cover the same
frame, the frame keeps the maximum score, matching the EVAL paper's projection
rule.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def video_frame_count(dataset_root: Path, video_id: str) -> int:
    video_path = dataset_root / "testing_videos" / f"{int(video_id):02d}.avi"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open test video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Avenue Dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/avenue_eval10_frame_scores.csv"))
    args = parser.parse_args()

    if not args.scores.exists() or args.scores.stat().st_size == 0:
        raise RuntimeError(f"Score file missing or empty: {args.scores}")

    scores = pd.read_csv(args.scores)
    if scores.empty:
        raise RuntimeError(f"No rows found in {args.scores}")

    rows = []
    for video_id, group in scores.groupby("video_id"):
        video_id_str = f"{int(video_id):02d}"
        frame_count = video_frame_count(args.dataset_root, video_id_str)
        frame_scores = np.zeros(frame_count, dtype=np.float32)

        for _, row in group.iterrows():
            start = max(1, int(row["start_frame"]))
            end = min(frame_count, int(row["end_frame"]))
            score = float(row["anomaly_score"])
            frame_scores[start - 1 : end] = np.maximum(frame_scores[start - 1 : end], score)

        for frame_idx, score in enumerate(frame_scores, start=1):
            rows.append(
                {
                    "video_id": video_id_str,
                    "frame": frame_idx,
                    "frame_score": float(score),
                }
            )
        print(f"video {video_id_str}: {frame_count} frame scores")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "frame", "frame_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} frame-score rows -> {args.out}")


if __name__ == "__main__":
    main()
