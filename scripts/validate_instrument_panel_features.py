"""Validate generated instrument-panel feature CSVs against source clip files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_instrument_panel_features import aggregate_panel_features, load_json, load_jsonl


def load_csv_by_clip(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["clip_id"]: row for row in reader}


def coerce(value: str):
    try:
        as_float = float(value)
    except ValueError:
        return value
    if as_float.is_integer():
        return int(as_float)
    return as_float


def coerce_expected(value: str, expected):
    if isinstance(expected, str):
        return value
    return coerce(value)


def almost_equal(left, right, tolerance: float) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return abs(float(left) - float(right)) <= tolerance


def validate(
    *,
    clip_manifest: Path,
    feature_csv: Path,
    train_csv: Path,
    test_csv: Path,
    report_out: Path,
    tolerance: float,
) -> dict:
    clips = load_jsonl(clip_manifest)
    features = load_csv_by_clip(feature_csv)
    train = load_csv_by_clip(train_csv)
    test = load_csv_by_clip(test_csv)
    warnings: list[str] = []
    mismatches: list[str] = []

    if len(features) != len(clips):
        warnings.append(f"Feature CSV row count {len(features)} != clip manifest count {len(clips)}")

    for clip in clips:
        clip_id = clip["clip_id"]
        if clip_id not in features:
            mismatches.append(f"{clip_id}: missing from all-features CSV")
            continue
        physics = load_json(Path(clip["physics_path"]))
        expected = aggregate_panel_features(clip, physics)
        raw_actual = features[clip_id]
        for key, expected_value in expected.items():
            if key not in raw_actual:
                mismatches.append(f"{clip_id}: missing column {key}")
                continue
            actual_value = coerce_expected(raw_actual[key], expected_value)
            if not almost_equal(actual_value, expected_value, tolerance):
                mismatches.append(
                    f"{clip_id}: {key} expected={expected_value!r} actual={actual_value!r}"
                )
                if len(mismatches) >= 25:
                    break

        direction_sum = sum(float(raw_actual.get(f"direction_hist_{idx:02d}", 0.0)) for idx in range(12))
        if int(float(raw_actual.get("track_count", 0))) > 0 and abs(direction_sum - 1.0) > 1e-4:
            warnings.append(f"{clip_id}: direction histogram sums to {direction_sum:.6f}")

        for idx in range(12):
            value = float(raw_actual.get(f"speed_ray_norm_{idx:02d}", 0.0))
            if value < -1e-6 or value > 1.0 + 1e-6:
                warnings.append(f"{clip_id}: speed_ray_norm_{idx:02d} out of range: {value}")

        if len(mismatches) >= 25:
            break

    train_bad = [clip_id for clip_id, row in train.items() if row.get("split") != "training" or int(float(row.get("clip_label", 1))) != 0]
    test_bad = [clip_id for clip_id, row in test.items() if row.get("split") != "testing"]
    if train_bad:
        warnings.append(f"Train CSV contains non-normal/non-training rows: {train_bad[:5]}")
    if test_bad:
        warnings.append(f"Test CSV contains non-testing rows: {test_bad[:5]}")

    payload = {
        "clip_manifest": str(clip_manifest),
        "feature_csv": str(feature_csv),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "clips_checked": len(clips),
        "feature_rows": len(features),
        "train_rows": len(train),
        "test_rows": len(test),
        "mismatch_count": len(mismatches),
        "warning_count": len(warnings),
        "mismatch_examples": mismatches[:25],
        "warning_examples": warnings[:25],
        "status": "passed" if not mismatches and not warnings else "needs_review",
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-manifest", type=Path, default=Path("manifests/anomalygpt_avenue_clips.jsonl"))
    parser.add_argument("--feature-csv", type=Path, default=Path("features/avenue_instrument_panel_features.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("features/avenue_instrument_panel_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("features/avenue_instrument_panel_test.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/avenue_instrument_panel_validation.json"))
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    payload = validate(
        clip_manifest=args.clip_manifest,
        feature_csv=args.feature_csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        report_out=args.report_out,
        tolerance=args.tolerance,
    )
    print(json.dumps({key: payload[key] for key in ["status", "clips_checked", "mismatch_count", "warning_count"]}, indent=2))
    print(f"Wrote: {args.report_out}")


if __name__ == "__main__":
    main()
