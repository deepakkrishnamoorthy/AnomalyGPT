"""Build an EVAL-style 10-frame spatial video-volume manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def video_info(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, width, height


def region_grid(width: int, height: int, region_size: int) -> list[dict]:
    stride = region_size // 2
    # Match EVAL's padding idea: pad to whole region multiples, then use half-overlap.
    pad_y = 0 if height % region_size == 0 else region_size - (height % region_size)
    pad_x = 0 if width % region_size == 0 else region_size - (width % region_size)
    padded_h = height + pad_y
    padded_w = width + pad_x

    regions = []
    idx = 0
    for y in range(0, padded_h - region_size + 1, stride):
        for x in range(0, padded_w - region_size + 1, stride):
            regions.append({"region_id": idx, "x": x, "y": y, "w": region_size, "h": region_size})
            idx += 1
    return regions


def build_manifest(dataset_root: Path, *, depth: int, region_size: int, temporal_stride: int) -> list[dict]:
    rows = []
    split_map = {
        "training": "training_videos",
        "testing": "testing_videos",
    }
    for split, folder in split_map.items():
        for video_path in sorted((dataset_root / folder).glob("*.avi")):
            frame_count, width, height = video_info(video_path)
            regions = region_grid(width, height, region_size)
            starts = list(range(1, max(frame_count - depth + 2, 1), temporal_stride))
            if starts and starts[-1] != frame_count - depth + 1 and frame_count >= depth:
                starts.append(frame_count - depth + 1)
            for start in starts:
                end = min(frame_count, start + depth - 1)
                for region in regions:
                    rows.append(
                        {
                            "dataset": "avenue",
                            "split": split,
                            "video_id": video_path.stem,
                            "video_path": str(video_path),
                            "volume_id": f"avenue_{split}_{video_path.stem}_t{start:05d}_r{region['region_id']:03d}",
                            "start_frame": start,
                            "end_frame": end,
                            "depth": end - start + 1,
                            "region_id": region["region_id"],
                            "x": region["x"],
                            "y": region["y"],
                            "w": region["w"],
                            "h": region["h"],
                            "source_width": width,
                            "source_height": height,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("Avenue Dataset"))
    parser.add_argument("--out", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--region-size", type=int, default=128)
    parser.add_argument("--temporal-stride", type=int, default=10, help="Use 1 for faithful dense EVAL; 10 is lighter.")
    args = parser.parse_args()

    rows = build_manifest(
        args.dataset_root,
        depth=args.depth,
        region_size=args.region_size,
        temporal_stride=args.temporal_stride,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} volume rows -> {args.out}")


if __name__ == "__main__":
    main()

