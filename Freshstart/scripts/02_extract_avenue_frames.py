"""Extract AVI videos into EVAL-compatible frame folders.

This is intentionally separate from auditing because it creates many files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_video(video_path: Path, out_dir: Path, *, overwrite: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if existing and not overwrite:
        return len(existing)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{count + 1:05d}.jpg"), frame)
        count += 1
    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("Avenue Dataset"))
    parser.add_argument("--out-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_map = {
        "training": ("training_videos", "Train"),
        "testing": ("testing_videos", "Test"),
    }
    selected = split_map if args.split == "all" else {args.split: split_map[args.split]}

    for split, (video_dir_name, eval_split_name) in selected.items():
        video_dir = args.dataset_root / video_dir_name
        for video_path in sorted(video_dir.glob("*.avi")):
            out_dir = args.out_root / eval_split_name / video_path.stem
            count = extract_video(video_path, out_dir, overwrite=args.overwrite)
            print(f"{split}/{video_path.name}: {count} frames -> {out_dir}")


if __name__ == "__main__":
    main()

