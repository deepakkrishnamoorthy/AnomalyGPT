"""Build 32-frame AnomalyGPT clip records from the Avenue manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.temporal_metrics import TemporalInterval, intervals_to_frame_labels


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def video_intervals(record: dict) -> list[TemporalInterval]:
    return [
        TemporalInterval(start=item["start"], end=item["end"])
        for item in record.get("anomaly_intervals", [])
    ]


def frame_path(record: dict, frame_id_1based: int) -> str:
    first_frame = record["first_frame"] or "0000.jpg"
    width = len(Path(first_frame).stem)
    frame_index_0based = frame_id_1based - 1
    return f"datasets/{record['frame_dir']}/{frame_index_0based:0{width}d}{record['frame_ext']}"


def window_starts(frame_count: int, clip_len: int, stride: int) -> list[int]:
    if frame_count < clip_len:
        return [1]
    starts = list(range(1, frame_count - clip_len + 2, stride))
    tail_start = frame_count - clip_len + 1
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def clipped_intervals(
    intervals: list[TemporalInterval],
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    clipped: list[dict] = []
    for interval in intervals:
        start = max(interval.start, start_frame)
        end = min(interval.end, end_frame)
        if start <= end:
            clipped.append(
                {
                    "start": start,
                    "end": end,
                    "local_start": start - start_frame + 1,
                    "local_end": end - start_frame + 1,
                }
            )
    return clipped


def build_qa(clip: dict) -> dict:
    if clip["clip_label"] == 0:
        answer = (
            "<assessment>No anomaly is present in this clip. "
            "The observed motion is treated as normal scene behavior.</assessment>"
            "<physics>Physics features are pending tracking; frame-level ground truth marks this clip normal.</physics>"
        )
        question = "What anomaly is present in this video clip?"
    else:
        answer = (
            "<assessment>An anomalous event is present in this clip according to the Avenue "
            "frame-level annotation.</assessment>"
            "<physics>Physics features are pending tracking; this clip overlaps the annotated "
            f"anomaly interval for {clip['anomaly_frame_count']} frame(s).</physics><SEG>"
        )
        question = "Explain why the highlighted behavior is abnormal in this video clip."

    return {
        "clip_id": clip["clip_id"],
        "question": question,
        "answer": answer,
        "clip_label": clip["clip_label"],
        "frame_labels": clip["frame_labels"],
        "anomaly_intervals": clip["anomaly_intervals"],
        "target_track_ids": [],
        "mask_frames": [],
        "physics_status": "pending_tracking",
    }


def build_placeholder_tracks(clip: dict) -> dict:
    return {
        "clip_id": clip["clip_id"],
        "dataset": "avenue",
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "pending_tracking",
        "tracks": [],
    }


def build_placeholder_physics(clip: dict) -> dict:
    return {
        "clip_id": clip["clip_id"],
        "dataset": "avenue",
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "pending_tracking",
        "features": [],
        "crowd_flow": None,
        "ground_truth": {
            "clip_label": clip["clip_label"],
            "anomaly_frame_count": clip["anomaly_frame_count"],
            "anomaly_overlap": clip["anomaly_overlap"],
        },
    }


def make_clip_record(
    *,
    record: dict,
    clip_index: int,
    start_frame: int,
    clip_len: int,
    fps: float,
    output_root: Path,
) -> dict:
    end_frame = min(record["frame_count"], start_frame + clip_len - 1)
    intervals = video_intervals(record)
    all_labels = intervals_to_frame_labels(intervals, record["frame_count"])
    frame_ids = list(range(start_frame, end_frame + 1))
    frame_labels = [all_labels[frame_id - 1] for frame_id in frame_ids]
    anomaly_frame_count = sum(frame_labels)
    clip_label = int(anomaly_frame_count > 0)
    clip_id = f"avenue_{record['split']}_{record['video_id']}_clip_{clip_index:06d}"
    clip_dir = output_root / clip_id
    anomaly_overlap = anomaly_frame_count / len(frame_ids)

    return {
        "dataset": "avenue",
        "clip_id": clip_id,
        "split": record["split"],
        "video_id": record["video_id"],
        "clip_index": clip_index,
        "fps": fps,
        "width": record["width"],
        "height": record["height"],
        "clip_len": len(frame_ids),
        "configured_clip_len": clip_len,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_ids": frame_ids,
        "frame_paths": [frame_path(record, frame_id) for frame_id in frame_ids],
        "frame_labels": frame_labels,
        "clip_label": clip_label,
        "anomaly_frame_count": anomaly_frame_count,
        "anomaly_overlap": round(anomaly_overlap, 6),
        "anomaly_intervals": clipped_intervals(intervals, start_frame, end_frame),
        "source_video_anomaly_intervals": [
            {"start": interval.start, "end": interval.end} for interval in intervals
        ],
        "clip_dir": clip_dir.as_posix(),
        "meta_path": (clip_dir / "meta.json").as_posix(),
        "qa_path": (clip_dir / "qa.json").as_posix(),
        "tracks_path": (clip_dir / "tracks.json").as_posix(),
        "physics_path": (clip_dir / "physics.json").as_posix(),
        "masks_dir": (clip_dir / "masks").as_posix(),
    }


def materialize_clip_files(clip: dict) -> None:
    clip_dir = Path(clip["clip_dir"])
    masks_dir = Path(clip["masks_dir"])
    masks_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        key: clip[key]
        for key in [
            "dataset",
            "clip_id",
            "split",
            "video_id",
            "clip_index",
            "fps",
            "width",
            "height",
            "clip_len",
            "configured_clip_len",
            "start_frame",
            "end_frame",
            "frame_ids",
            "frame_paths",
            "frame_labels",
            "clip_label",
            "anomaly_frame_count",
            "anomaly_overlap",
            "anomaly_intervals",
            "source_video_anomaly_intervals",
        ]
    }
    write_json(Path(clip["meta_path"]), meta)
    write_json(Path(clip["qa_path"]), build_qa(clip))
    write_json(Path(clip["tracks_path"]), build_placeholder_tracks(clip))
    write_json(Path(clip["physics_path"]), build_placeholder_physics(clip))
    (masks_dir / ".gitkeep").write_text("", encoding="utf-8")


def validate_clips(clips: list[dict], manifest: dict) -> list[str]:
    warnings: list[str] = []
    by_video: dict[tuple[str, str], list[dict]] = {}
    for clip in clips:
        by_video.setdefault((clip["split"], clip["video_id"]), []).append(clip)
        if clip["clip_label"] != int(any(clip["frame_labels"])):
            warnings.append(f"{clip['clip_id']}: clip_label disagrees with frame_labels")
        if clip["anomaly_frame_count"] != sum(clip["frame_labels"]):
            warnings.append(f"{clip['clip_id']}: anomaly_frame_count disagrees with frame_labels")
        if clip["clip_len"] != len(clip["frame_paths"]):
            warnings.append(f"{clip['clip_id']}: clip_len disagrees with frame_paths")

    records = {(item["split"], item["video_id"]): item for item in manifest["videos"]}
    for key, video_clips in by_video.items():
        source = records[key]
        first_start = min(item["start_frame"] for item in video_clips)
        last_end = max(item["end_frame"] for item in video_clips)
        if first_start != 1:
            warnings.append(f"{key}: clips do not start at frame 1")
        if last_end != source["frame_count"]:
            warnings.append(f"{key}: clips end at {last_end}, expected {source['frame_count']}")
    return warnings


def summarize(clips: list[dict], warnings: list[str], *, clip_len: int, train_stride: int, test_stride: int) -> str:
    train = [clip for clip in clips if clip["split"] == "training"]
    test = [clip for clip in clips if clip["split"] == "testing"]
    positive = [clip for clip in clips if clip["clip_label"] == 1]
    test_positive = [clip for clip in test if clip["clip_label"] == 1]
    total_clip_frames = sum(clip["clip_len"] for clip in clips)
    total_positive_clip_frames = sum(clip["anomaly_frame_count"] for clip in clips)

    lines = [
        "# Avenue Clip Dataset Validation",
        "",
        "## Windowing",
        "",
        f"- Clip length: {clip_len}",
        f"- Training stride: {train_stride}",
        f"- Testing stride: {test_stride}",
        "",
        "## Summary",
        "",
        f"- Total clips: {len(clips)}",
        f"- Training clips: {len(train)}",
        f"- Testing clips: {len(test)}",
        f"- Positive clips: {len(positive)}",
        f"- Positive testing clips: {len(test_positive)}",
        f"- Total clip-frame observations: {total_clip_frames}",
        f"- Positive frame labels inside clips: {total_positive_clip_frames}",
        "",
        "Because clips overlap, positive frame labels inside clips are expected to exceed the unique anomalous frame count in `avenue_eda.md`.",
        "",
        "## Per-Video Clip Counts",
        "",
        "| Split | Video | Clips | Positive Clips | Mean Overlap |",
        "|---|---:|---:|---:|---:|",
    ]

    keys = sorted({(clip["split"], clip["video_id"]) for clip in clips})
    for split, video_id in keys:
        group = [clip for clip in clips if clip["split"] == split and clip["video_id"] == video_id]
        positives = [clip for clip in group if clip["clip_label"] == 1]
        mean_overlap = sum(clip["anomaly_overlap"] for clip in group) / len(group)
        lines.append(
            f"| {split} | {video_id} | {len(group)} | {len(positives)} | {mean_overlap:.4f} |"
        )

    lines.extend(["", "## Validation Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None. Clip labels, frame labels, and video coverage checks passed.")
    lines.append("")
    return "\n".join(lines)


def build_dataset(
    *,
    manifest_path: Path,
    output_root: Path,
    clip_manifest_out: Path,
    ground_truth_out: Path,
    report_out: Path,
    clip_len: int,
    train_stride: int,
    test_stride: int,
    fps: float,
    materialize: bool,
) -> tuple[list[dict], list[str]]:
    manifest = load_json(manifest_path)
    clips: list[dict] = []

    for record in manifest["videos"]:
        stride = train_stride if record["split"] == "training" else test_stride
        for clip_index, start_frame in enumerate(window_starts(record["frame_count"], clip_len, stride)):
            clip = make_clip_record(
                record=record,
                clip_index=clip_index,
                start_frame=start_frame,
                clip_len=clip_len,
                fps=fps,
                output_root=output_root,
            )
            clips.append(clip)
            if materialize:
                materialize_clip_files(clip)

    warnings = validate_clips(clips, manifest)
    write_jsonl(clip_manifest_out, clips)
    write_json(
        ground_truth_out,
        {
            "dataset": "avenue",
            "clip_len": clip_len,
            "train_stride": train_stride,
            "test_stride": test_stride,
            "fps": fps,
            "frame_indexing": "1_based_inclusive",
            "clips": [
                {
                    "clip_id": clip["clip_id"],
                    "split": clip["split"],
                    "video_id": clip["video_id"],
                    "start_frame": clip["start_frame"],
                    "end_frame": clip["end_frame"],
                    "frame_ids": clip["frame_ids"],
                    "frame_labels": clip["frame_labels"],
                    "clip_label": clip["clip_label"],
                    "anomaly_frame_count": clip["anomaly_frame_count"],
                    "anomaly_overlap": clip["anomaly_overlap"],
                    "anomaly_intervals": clip["anomaly_intervals"],
                }
                for clip in clips
            ],
        },
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        summarize(
            clips,
            warnings,
            clip_len=clip_len,
            train_stride=train_stride,
            test_stride=test_stride,
        ),
        encoding="utf-8",
    )
    return clips, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/anomalygpt/avenue_clips"))
    parser.add_argument("--clip-manifest-out", type=Path, default=Path("manifests/anomalygpt_avenue_clips.jsonl"))
    parser.add_argument("--ground-truth-out", type=Path, default=Path("manifests/avenue_clip_ground_truth.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/avenue_clip_dataset_validation.md"))
    parser.add_argument("--clip-len", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=16)
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--no-materialize", action="store_true")
    args = parser.parse_args()

    clips, warnings = build_dataset(
        manifest_path=args.manifest,
        output_root=args.output_root,
        clip_manifest_out=args.clip_manifest_out,
        ground_truth_out=args.ground_truth_out,
        report_out=args.report_out,
        clip_len=args.clip_len,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
        fps=args.fps,
        materialize=not args.no_materialize,
    )
    print(f"Built {len(clips)} clips")
    print(f"Warnings: {len(warnings)}")
    print(f"Wrote: {args.clip_manifest_out}")
    print(f"Wrote: {args.ground_truth_out}")
    print(f"Wrote: {args.report_out}")


if __name__ == "__main__":
    main()
