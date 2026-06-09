"""Render Avenue frames with frame-level anomaly annotations overlaid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def is_anomalous(frame_id_1based: int, intervals: list[dict]) -> bool:
    return any(item["start"] <= frame_id_1based <= item["end"] for item in intervals)


def find_record(manifest: dict, split: str, video_id: str) -> dict:
    for record in manifest["videos"]:
        if record["split"] == split and record["video_id"] == video_id:
            return record
    raise ValueError(f"No record found for split={split!r}, video_id={video_id!r}")


def frame_path(dataset_root: Path, record: dict, frame_index_0based: int) -> Path:
    frame_dir = dataset_root / record["frame_dir"]
    first_name = record["first_frame"] or "0000.jpg"
    width = len(Path(first_name).stem)
    return frame_dir / f"{frame_index_0based:0{width}d}{record['frame_ext']}"


def draw_overlay(frame, *, record: dict, frame_index_0based: int, anomaly: bool):
    frame_id = frame_index_0based + 1
    color = (0, 0, 255) if anomaly else (0, 180, 0)
    label = "ANOMALY" if anomaly else "NORMAL"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        f"Avenue {record['split']} video {record['video_id']} | frame {frame_id}/{record['frame_count']} | {label}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )
    intervals_text = ", ".join(
        f"{item['start']}-{item['end']}" for item in record["anomaly_intervals"]
    )
    if intervals_text:
        cv2.putText(
            frame,
            f"GT intervals: {intervals_text}",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    if anomaly:
        cv2.rectangle(frame, (4, 4), (frame.shape[1] - 5, frame.shape[0] - 5), color, 5)
    return frame


def render_video(
    *,
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    video_id: str,
    output_path: Path,
    fps: float,
    start_frame: int | None,
    end_frame: int | None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = find_record(manifest, split, video_id)

    first = max(1, start_frame or 1)
    last = min(record["frame_count"], end_frame or record["frame_count"])
    if first > last:
        raise ValueError(f"Invalid frame range: {first}-{last}")

    first_path = frame_path(dataset_root, record, first - 1)
    first_frame = cv2.imread(str(first_path))
    if first_frame is None:
        raise FileNotFoundError(first_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    written = 0
    try:
        for frame_id in range(first, last + 1):
            path = frame_path(dataset_root, record, frame_id - 1)
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(path)
            frame = draw_overlay(
                frame,
                record=record,
                frame_index_0based=frame_id - 1,
                anomaly=is_anomalous(frame_id, record["anomaly_intervals"]),
            )
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"Wrote {output_path} ({written} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--split", choices=["training", "testing"], default="testing")
    parser.add_argument("--video-id", default="01")
    parser.add_argument("--output", type=Path, default=Path("outputs/avenue_testing_01_annotated.mp4"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start-frame", type=int, default=None, help="1-based inclusive frame id")
    parser.add_argument("--end-frame", type=int, default=None, help="1-based inclusive frame id")
    args = parser.parse_args()

    render_video(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        split=args.split,
        video_id=args.video_id,
        output_path=args.output,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
