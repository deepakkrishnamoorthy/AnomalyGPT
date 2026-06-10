"""Export EVAL-style spatial video volumes for visual validation.

Reads the JSONL manifest produced by 03_build_eval_volume_manifest.py and
materializes each 10-frame spatial region as:

- a compressed grayscale .npz array for later model/debug use
- a contact-sheet .jpg preview for quick visual validation
- an exported-volume manifest with paths to both artifacts
"""

from __future__ import annotations

import argparse
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


def read_crop(frame_path: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
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
            value=(0, 0, 0),
        )

    crop_bgr = frame[y : y + h, x : x + w]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return crop_gray, crop_bgr


def load_volume(row: dict, frames_root: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    split_dir = SPLIT_TO_FRAME_DIR[row["split"]]
    video_dir = frames_root / split_dir / row["video_id"]
    gray_frames = []
    preview_frames = []

    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame_path = video_dir / f"{frame_idx:05d}.jpg"
        gray, color = read_crop(frame_path, row)
        gray_frames.append(gray)
        preview_frames.append(color)

    return np.stack(gray_frames, axis=0), preview_frames


def save_contact_sheet(frames: list[np.ndarray], out_path: Path, *, tile_width: int) -> None:
    tiles = []
    for idx, frame in enumerate(frames, start=1):
        tile = frame.copy()
        cv2.putText(
            tile,
            str(idx),
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    sheet = np.concatenate(tiles, axis=1)
    if tile_width and frames[0].shape[1] != tile_width:
        scale = tile_width / frames[0].shape[1]
        sheet = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def artifact_paths(out_root: Path, row: dict) -> tuple[Path, Path]:
    split = row["split"]
    video_id = row["video_id"]
    region_id = f"r{int(row['region_id']):03d}"
    start = f"t{int(row['start_frame']):05d}"
    stem = row["volume_id"]
    npz_path = out_root / "npz" / split / video_id / region_id / f"{stem}.npz"
    preview_path = out_root / "previews" / split / video_id / region_id / f"{stem}.jpg"
    return npz_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-root", type=Path, default=Path("data/avenue_eval10_volumes"))
    parser.add_argument("--index-out", type=Path, default=Path("manifests/avenue_eval10_saved_volumes_manifest.jsonl"))
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N selected volumes for validation.")
    parser.add_argument("--preview-tile-width", type=int, default=96)
    parser.add_argument("--skip-existing", action="store_true", help="Skip volume rows whose npz and preview already exist.")
    parser.add_argument("--no-npz", action="store_true", help="Only save preview contact sheets.")
    parser.add_argument("--no-preview", action="store_true", help="Only save npz volumes.")
    args = parser.parse_args()

    if args.no_npz and args.no_preview:
        raise ValueError("At least one output type must be enabled.")

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    selected = 0

    with args.index_out.open("w", encoding="utf-8") as index_handle:
        for row in iter_manifest(args.manifest):
            if args.split != "all" and row["split"] != args.split:
                continue
            if args.limit is not None and selected >= args.limit:
                break
            selected += 1

            npz_path, preview_path = artifact_paths(args.out_root, row)
            npz_done = args.no_npz or npz_path.exists()
            preview_done = args.no_preview or preview_path.exists()
            if args.skip_existing and npz_done and preview_done:
                out_row = dict(row)
                if not args.no_npz:
                    out_row["npz_path"] = str(npz_path)
                if not args.no_preview:
                    out_row["preview_path"] = str(preview_path)
                index_handle.write(json.dumps(out_row, sort_keys=True) + "\n")
                continue

            volume, preview_frames = load_volume(row, args.frames_root)

            if not args.no_npz:
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    volume=volume,
                    volume_id=row["volume_id"],
                    split=row["split"],
                    video_id=row["video_id"],
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    region_id=int(row["region_id"]),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                )

            if not args.no_preview:
                save_contact_sheet(preview_frames, preview_path, tile_width=args.preview_tile_width)

            out_row = dict(row)
            if not args.no_npz:
                out_row["npz_path"] = str(npz_path)
            if not args.no_preview:
                out_row["preview_path"] = str(preview_path)
            index_handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            exported += 1

            if exported % 1000 == 0:
                print(f"Exported {exported} volumes...")

    print(f"Selected {selected} volumes. Newly exported {exported}. Index -> {args.index_out}")


if __name__ == "__main__":
    main()
