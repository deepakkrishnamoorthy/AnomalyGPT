"""Build instrument-panel feature tables from Avenue clip physics files.

The resulting rows are designed for simple anomaly baselines such as
autoencoders, one-class models, and nearest-normal exemplar retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


OBJECT_CLASS_COLUMNS = [
    "object_person",
    "object_car",
    "object_cyclist",
    "object_dog",
    "object_tree",
    "object_house",
    "object_skyscraper",
    "object_bridge",
]

NUM_DIRECTION_BINS = 12

TRACK_CATEGORY_TO_OBJECT_COLUMN = {
    "person": "object_person",
    "car": "object_car",
    "bicycle": "object_cyclist",
    "motorcycle": "object_cyclist",
    "cyclist": "object_cyclist",
    "dog": "object_dog",
    "tree": "object_tree",
    "house": "object_house",
    "skyscraper": "object_skyscraper",
    "bridge": "object_bridge",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def safe_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def safe_min(values: list[float]) -> float:
    return min(values) if values else 0.0


def direction_bin(direction_deg: float) -> int:
    normalized = direction_deg % 360.0
    return int(normalized // (360.0 / NUM_DIRECTION_BINS)) % NUM_DIRECTION_BINS


def normalize_histogram(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def prefixed_stats(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": safe_mean(values),
        f"{prefix}_median": safe_median(values),
        f"{prefix}_std": safe_std(values),
        f"{prefix}_min": safe_min(values),
        f"{prefix}_max": safe_max(values),
    }


def object_class_bars(clip: dict) -> dict[str, float]:
    tracks_path = Path(clip["tracks_path"])
    if not tracks_path.exists():
        return {column: 0.0 for column in OBJECT_CLASS_COLUMNS}
    tracks_payload = load_json(tracks_path)
    counts = {column: 0 for column in OBJECT_CLASS_COLUMNS}
    total = 0
    for track in tracks_payload.get("tracks", []):
        category = str(track.get("panel_class") or track.get("category") or "").lower()
        column = TRACK_CATEGORY_TO_OBJECT_COLUMN.get(category)
        if column is None:
            continue
        counts[column] += max(len(track.get("boxes", [])), 1)
        total += max(len(track.get("boxes", [])), 1)
    if total <= 0:
        return {column: 0.0 for column in OBJECT_CLASS_COLUMNS}
    return {column: counts[column] / total for column in OBJECT_CLASS_COLUMNS}


def aggregate_panel_features(clip: dict, physics: dict) -> dict:
    features = physics.get("features", [])
    track_count = len(features)
    candidate_count = sum(1 for item in features if item.get("is_candidate_anomaly"))
    overlap_count = sum(1 for item in features if item.get("overlaps_anomaly_frames"))

    track_lengths = [float(item.get("track_length", 0.0)) for item in features]
    velocities = [float(item.get("mean_velocity_px_s", 0.0)) for item in features]
    max_velocities = [float(item.get("max_velocity_px_s", 0.0)) for item in features]
    accelerations = [float(item.get("mean_acceleration_px_s2", 0.0)) for item in features]
    curvatures = [float(item.get("curvature_deg", 0.0)) for item in features]
    stationary = [float(item.get("stationary_fraction", 0.0)) for item in features]
    velocity_ratios = [float(item.get("velocity_ratio", 0.0)) for item in features]
    flow_deviations = [float(item.get("flow_deviation_deg", 0.0)) for item in features]

    direction_weights = [0.0] * NUM_DIRECTION_BINS
    speed_sums = [0.0] * NUM_DIRECTION_BINS
    speed_counts = [0] * NUM_DIRECTION_BINS
    for item in features:
        bin_id = direction_bin(float(item.get("direction_deg", 0.0)))
        weight = max(float(item.get("track_length", 0.0)), 1.0)
        speed = float(item.get("mean_velocity_px_s", 0.0))
        direction_weights[bin_id] += weight
        speed_sums[bin_id] += speed
        speed_counts[bin_id] += 1

    direction_hist = normalize_histogram(direction_weights)
    speed_by_direction = [
        speed_sums[idx] / speed_counts[idx] if speed_counts[idx] else 0.0
        for idx in range(NUM_DIRECTION_BINS)
    ]
    max_speed_ray = max(speed_by_direction) if speed_by_direction else 0.0
    speed_by_direction_norm = [
        value / max_speed_ray if max_speed_ray > 0 else 0.0
        for value in speed_by_direction
    ]

    crowd_flow = physics.get("crowd_flow") or {}
    row: dict[str, str | int | float] = {
        "clip_id": clip["clip_id"],
        "dataset": clip["dataset"],
        "split": clip["split"],
        "video_id": clip["video_id"],
        "clip_index": clip["clip_index"],
        "start_frame": clip["start_frame"],
        "end_frame": clip["end_frame"],
        "clip_len": clip["clip_len"],
        "clip_label": clip["clip_label"],
        "anomaly_frame_count": clip["anomaly_frame_count"],
        "anomaly_overlap": clip["anomaly_overlap"],
        "track_count": track_count,
        "candidate_track_count": candidate_count,
        "candidate_track_fraction": candidate_count / track_count if track_count else 0.0,
        "gt_overlap_track_count": overlap_count,
        "gt_overlap_track_fraction": overlap_count / track_count if track_count else 0.0,
        "stationary_panel_fraction": safe_mean(stationary) if features else 1.0,
        "moving_panel_fraction": 1.0 - (safe_mean(stationary) if features else 1.0),
        "no_motion_indicator": 1 if track_count == 0 else 0,
        "object_unknown_motion": 1 if track_count > 0 else 0,
        "crowd_mean_speed_px_s": float(crowd_flow.get("mean_speed_px_s", 0.0)),
        "crowd_mean_direction_deg": float(crowd_flow.get("mean_direction_deg", 0.0)),
    }

    row.update(object_class_bars(clip))

    for prefix, values in [
        ("track_length", track_lengths),
        ("velocity", velocities),
        ("max_velocity", max_velocities),
        ("acceleration", accelerations),
        ("curvature", curvatures),
        ("stationary", stationary),
        ("velocity_ratio", velocity_ratios),
        ("flow_deviation", flow_deviations),
    ]:
        row.update(prefixed_stats(prefix, values))

    for idx, value in enumerate(direction_hist):
        row[f"direction_hist_{idx:02d}"] = value
    for idx, value in enumerate(speed_by_direction):
        row[f"speed_ray_{idx:02d}"] = value
    for idx, value in enumerate(speed_by_direction_norm):
        row[f"speed_ray_norm_{idx:02d}"] = value

    return {key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()}


def feature_columns(rows: list[dict]) -> list[str]:
    leading = [
        "clip_id",
        "dataset",
        "split",
        "video_id",
        "clip_index",
        "start_frame",
        "end_frame",
        "clip_len",
        "clip_label",
        "anomaly_frame_count",
        "anomaly_overlap",
    ]
    remaining = sorted(key for key in rows[0].keys() if key not in leading)
    return leading + remaining


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def numeric_feature_columns(columns: list[str]) -> list[str]:
    excluded = {
        "clip_id",
        "dataset",
        "split",
        "video_id",
        "clip_index",
        "start_frame",
        "end_frame",
        "clip_len",
        "clip_label",
        "anomaly_frame_count",
        "anomaly_overlap",
    }
    return [column for column in columns if column not in excluded]


def render_report(rows: list[dict], columns: list[str]) -> str:
    train = [row for row in rows if row["split"] == "training"]
    test = [row for row in rows if row["split"] == "testing"]
    positives = [row for row in rows if row["clip_label"] == 1]
    numeric_cols = numeric_feature_columns(columns)

    lines = [
        "# Avenue Instrument Panel Feature EDA",
        "",
        "## Summary",
        "",
        f"- Total clips: {len(rows)}",
        f"- Training clips: {len(train)}",
        f"- Testing clips: {len(test)}",
        f"- Positive clips: {len(positives)}",
        f"- Numeric feature columns: {len(numeric_cols)}",
        "",
        "## Feature Groups",
        "",
        "- Object class bars: currently placeholder columns because no object classifier is integrated yet.",
        "- Stationary/moving panel: aggregated from pseudo-track stationary fractions.",
        "- Direction histogram: 12 bins weighted by pseudo-track length.",
        "- Speed rays: mean pseudo-track speed per direction bin.",
        "- Physics stats: velocity, acceleration, curvature, flow deviation, velocity ratio, track length.",
        "",
        "## Selected Feature Means",
        "",
        "| Feature | Train Mean | Test Mean | Positive Mean |",
        "|---|---:|---:|---:|",
    ]

    selected = [
        "track_count",
        "candidate_track_fraction",
        "velocity_mean",
        "max_velocity_max",
        "acceleration_mean",
        "curvature_mean",
        "flow_deviation_mean",
        "stationary_panel_fraction",
        "moving_panel_fraction",
        "crowd_mean_speed_px_s",
    ]
    for column in selected:
        train_mean = safe_mean([float(row[column]) for row in train])
        test_mean = safe_mean([float(row[column]) for row in test])
        pos_mean = safe_mean([float(row[column]) for row in positives])
        lines.append(f"| {column} | {train_mean:.4f} | {test_mean:.4f} | {pos_mean:.4f} |")

    lines.extend(
        [
            "",
            "## Autoencoder Usage",
            "",
            "Use `avenue_instrument_panel_train.csv` for normal-only training.",
            "Use `avenue_instrument_panel_test.csv` for anomaly scoring and AUC evaluation.",
            "",
            "Recommended first feature set: drop metadata and labels, then standardize the numeric instrument-panel columns.",
            "",
        ]
    )
    return "\n".join(lines)


def build_tables(
    *,
    clip_manifest: Path,
    output_all: Path,
    output_train: Path,
    output_test: Path,
    report_out: Path,
) -> tuple[int, int, int]:
    clips = load_jsonl(clip_manifest)
    rows: list[dict] = []
    for clip in clips:
        physics_path = Path(clip["physics_path"])
        if not physics_path.exists():
            raise FileNotFoundError(f"Missing physics file for {clip['clip_id']}: {physics_path}")
        physics = load_json(physics_path)
        rows.append(aggregate_panel_features(clip, physics))

    columns = feature_columns(rows)
    train = [row for row in rows if row["split"] == "training" and row["clip_label"] == 0]
    test = [row for row in rows if row["split"] == "testing"]

    write_csv(output_all, rows, columns)
    write_csv(output_train, train, columns)
    write_csv(output_test, test, columns)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(render_report(rows, columns), encoding="utf-8")
    return len(rows), len(train), len(test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-manifest", type=Path, default=Path("manifests/anomalygpt_avenue_clips.jsonl"))
    parser.add_argument("--output-all", type=Path, default=Path("features/avenue_instrument_panel_features.csv"))
    parser.add_argument("--output-train", type=Path, default=Path("features/avenue_instrument_panel_train.csv"))
    parser.add_argument("--output-test", type=Path, default=Path("features/avenue_instrument_panel_test.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/avenue_instrument_panel_feature_eda.md"))
    args = parser.parse_args()

    total, train, test = build_tables(
        clip_manifest=args.clip_manifest,
        output_all=args.output_all,
        output_train=args.output_train,
        output_test=args.output_test,
        report_out=args.report_out,
    )
    print(f"Wrote rows: total={total}, train={train}, test={test}")
    print(f"Wrote: {args.output_all}")
    print(f"Wrote: {args.output_train}")
    print(f"Wrote: {args.output_test}")
    print(f"Wrote: {args.report_out}")


if __name__ == "__main__":
    main()
