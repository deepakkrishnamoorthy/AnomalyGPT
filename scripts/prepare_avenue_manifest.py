"""Create an Avenue dataset manifest and EDA report.

The Avenue ground-truth MAT file stores frame-level anomaly intervals for the
21 testing videos. This script keeps the manifest compact by recording frame
directories, frame naming metadata, and anomaly ranges rather than every image
path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.temporal_metrics import (
    TemporalInterval as Interval,
    interval_union_length,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - EDA can still run without dimensions.
    Image = None


@dataclass(frozen=True)
class VideoRecord:
    dataset: str
    split: str
    video_id: str
    frame_dir: str
    frame_ext: str
    frame_count: int
    first_frame: str | None
    last_frame: str | None
    width: int | None
    height: int | None
    anomaly_intervals: list[Interval]
    anomaly_frame_count: int
    anomaly_fraction: float


def normalize_gt_entry(entry) -> list[Interval]:
    """Normalize a MAT gt cell into inclusive 1-based anomaly intervals."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required by scipy for MAT parsing") from exc

    arr = np.asarray(entry).astype(int)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"Expected a 2-value interval, got shape {arr.shape}")
        starts = [int(arr[0])]
        ends = [int(arr[1])]
    elif arr.ndim == 2 and arr.shape[0] == 2:
        starts = [int(v) for v in arr[0].tolist()]
        ends = [int(v) for v in arr[1].tolist()]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        starts = [int(v) for v in arr[:, 0].tolist()]
        ends = [int(v) for v in arr[:, 1].tolist()]
    else:
        raise ValueError(f"Unsupported gt interval shape {arr.shape}")

    return [
        Interval(start=min(start, end), end=max(start, end))
        for start, end in zip(starts, ends)
    ]


def frame_files(frame_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def image_size(path: Path) -> tuple[int | None, int | None]:
    if Image is None or path is None:
        return None, None
    with Image.open(path) as img:
        return img.size


def load_test_gt(mat_path: Path) -> list[list[Interval]]:
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    if "gt" not in mat:
        raise KeyError(f"{mat_path} does not contain a 'gt' variable")
    return [normalize_gt_entry(entry) for entry in mat["gt"].ravel()]


def build_video_record(
    *,
    avenue_root: Path,
    split: str,
    video_dir: Path,
    intervals: list[Interval] | None = None,
) -> VideoRecord:
    files = frame_files(video_dir)
    if not files:
        raise ValueError(f"No frame files found in {video_dir}")
    width, height = image_size(files[0])
    intervals = intervals or []
    frame_count = len(files)
    anomaly_frame_count = interval_union_length(intervals)
    return VideoRecord(
        dataset="avenue",
        split=split,
        video_id=video_dir.name,
        frame_dir=video_dir.relative_to(avenue_root.parent).as_posix(),
        frame_ext=files[0].suffix.lower(),
        frame_count=frame_count,
        first_frame=files[0].name,
        last_frame=files[-1].name,
        width=width,
        height=height,
        anomaly_intervals=intervals,
        anomaly_frame_count=anomaly_frame_count,
        anomaly_fraction=round(anomaly_frame_count / frame_count, 6),
    )


def validate_records(records: list[VideoRecord]) -> list[str]:
    warnings: list[str] = []
    for record in records:
        for interval in record.anomaly_intervals:
            if interval.start < 1:
                warnings.append(f"{record.split}/{record.video_id}: interval starts before frame 1")
            if interval.end > record.frame_count:
                warnings.append(
                    f"{record.split}/{record.video_id}: interval {interval.start}-{interval.end} "
                    f"exceeds frame_count={record.frame_count}"
                )
    return warnings


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_report(manifest: dict, warnings: list[str]) -> str:
    records = manifest["videos"]
    train = [item for item in records if item["split"] == "training"]
    test = [item for item in records if item["split"] == "testing"]
    anomalous_test = [item for item in test if item["anomaly_frame_count"] > 0]
    total_frames = sum(item["frame_count"] for item in records)
    total_train_frames = sum(item["frame_count"] for item in train)
    total_test_frames = sum(item["frame_count"] for item in test)
    total_anomaly_frames = sum(item["anomaly_frame_count"] for item in test)
    total_intervals = sum(len(item["anomaly_intervals"]) for item in test)

    lines = [
        "# Avenue Dataset EDA",
        "",
        "## Summary",
        "",
        f"- Training videos: {len(train)}",
        f"- Testing videos: {len(test)}",
        f"- Total frames: {total_frames}",
        f"- Training frames: {total_train_frames}",
        f"- Testing frames: {total_test_frames}",
        f"- Test videos with anomalies: {len(anomalous_test)}",
        f"- Ground-truth anomaly intervals: {total_intervals}",
        f"- Ground-truth anomalous test frames: {total_anomaly_frames}",
        f"- Test anomaly frame fraction: {total_anomaly_frames / total_test_frames:.4f}",
        "",
        "## Annotation Format",
        "",
        "`avenue.mat` contains one variable, `gt`, with 21 entries. Each entry stores inclusive 1-based frame intervals for the corresponding testing video.",
        "",
        "Example interpretation:",
        "",
        "```text",
        "testing video 01: [78,120], [392,422], ...",
        "```",
        "",
        "These are frame-level labels. Pixel-level IoU/Dice requires separate masks or generated weak masks.",
        "",
        "## Evaluation Notes",
        "",
        "- Frame-level anomaly detection should use frame scores against the binary labels induced by these intervals.",
        "- Frame AUC is the first reliable metric available from this annotation file.",
        "- Interval IoU can compare predicted anomalous frame ranges with ground-truth ranges after thresholding frame scores.",
        "- Pixel IoU / Dice can be added after we create or obtain anomaly masks.",
        "",
        "Interval IoU definition:",
        "",
        "```text",
        "IoU = anomalous_frame_intersection / anomalous_frame_union",
        "```",
        "",
        "## Testing Videos",
        "",
        "| Video | Frames | Intervals | Anomaly Frames | Fraction |",
        "|---:|---:|---|---:|---:|",
    ]

    for item in test:
        intervals = ", ".join(
            f"{interval['start']}-{interval['end']}"
            for interval in item["anomaly_intervals"]
        )
        lines.append(
            f"| {item['video_id']} | {item['frame_count']} | {intervals or '-'} | "
            f"{item['anomaly_frame_count']} | {item['anomaly_fraction']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Training Videos",
            "",
            "| Video | Frames | Resolution |",
            "|---:|---:|---|",
        ]
    )
    for item in train:
        resolution = f"{item['width']}x{item['height']}" if item["width"] else "unknown"
        lines.append(f"| {item['video_id']} | {item['frame_count']} | {resolution} |")

    lines.extend(["", "## Validation Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None. All annotated intervals fit within available testing frame counts.")

    lines.append("")
    return "\n".join(lines)


def build_manifest(avenue_root: Path) -> tuple[dict, list[str]]:
    mat_path = avenue_root / "avenue.mat"
    train_root = avenue_root / "training" / "frames"
    test_root = avenue_root / "testing" / "frames"

    gt = load_test_gt(mat_path)
    train_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    test_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())

    if len(gt) != len(test_dirs):
        raise ValueError(f"gt entries ({len(gt)}) != testing videos ({len(test_dirs)})")

    records: list[VideoRecord] = []
    records.extend(
        build_video_record(
            avenue_root=avenue_root,
            split="training",
            video_dir=video_dir,
        )
        for video_dir in train_dirs
    )
    records.extend(
        build_video_record(
            avenue_root=avenue_root,
            split="testing",
            video_dir=video_dir,
            intervals=gt[index],
        )
        for index, video_dir in enumerate(test_dirs)
    )

    warnings = validate_records(records)
    manifest = {
        "dataset": "avenue",
        "root": avenue_root.as_posix(),
        "annotation_file": mat_path.relative_to(avenue_root.parent).as_posix(),
        "frame_indexing": "ground_truth_intervals_are_inclusive_1_based",
        "videos": [
            {
                **asdict(record),
                "anomaly_intervals": [asdict(interval) for interval in record.anomaly_intervals],
            }
            for record in records
        ],
    }
    return manifest, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avenue-root", type=Path, default=Path("datasets/avenue"))
    parser.add_argument("--manifest-out", type=Path, default=Path("manifests/avenue_manifest.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/avenue_eda.md"))
    args = parser.parse_args()

    manifest, warnings = build_manifest(args.avenue_root)
    write_json(args.manifest_out, manifest)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(manifest, warnings), encoding="utf-8")

    print(f"Wrote manifest: {args.manifest_out}")
    print(f"Wrote report: {args.report_out}")
    print(f"Videos: {len(manifest['videos'])}")
    print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
