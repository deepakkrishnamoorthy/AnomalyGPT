"""Fuse stride-10 exemplar anomaly scores with volume LEJEPA prediction errors.

The two score sources live on very different numeric scales, so this script
first applies an unsupervised robust normalization and then computes:

    fused = exemplar_weight * exemplar_norm + (1 - exemplar_weight) * lejepa_norm

No labels or ground truth are used by this script. It only joins per-volume
scores and creates a CSV compatible with the existing frame projection and
evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["volume_id"]
METADATA_COLUMNS = [
    "volume_id",
    "split",
    "video_id",
    "start_frame",
    "end_frame",
    "region_id",
    "x",
    "y",
    "w",
    "h",
]
OUTPUT_COLUMNS = [
    *METADATA_COLUMNS,
    "anomaly_score",
    "distance_app",
    "distance_ang",
    "distance_speed",
    "distance_bkg",
    "nearest_exemplar_volume_id",
    "exemplar_score_raw",
    "lejepa_score_raw",
    "exemplar_score_norm",
    "lejepa_score_norm",
    "fusion_exemplar_weight",
    "fusion_lejepa_weight",
]


def robust_scale(values: pd.Series, *, floor_zero: bool, clip_high: float | None) -> tuple[pd.Series, dict]:
    arr = values.astype(float)
    q50 = float(arr.quantile(0.50))
    q95 = float(arr.quantile(0.95))
    denom = max(q95 - q50, 1e-12)
    scaled = (arr - q50) / denom
    if floor_zero:
        scaled = scaled.clip(lower=0.0)
    if clip_high is not None:
        scaled = scaled.clip(upper=clip_high)
    return scaled, {"median": q50, "q95": q95, "denom": denom, "floor_zero": floor_zero, "clip_high": clip_high}


def load_scores(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{source_name} score file is missing or empty: {path}")
    df = pd.read_csv(path)
    missing = [col for col in [*METADATA_COLUMNS, "anomaly_score"] if col not in df.columns]
    if missing:
        raise RuntimeError(f"{source_name} score file is missing columns {missing}: {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exemplar-scores", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument(
        "--lejepa-scores",
        type=Path,
        default=Path("experiments/lejepa_volume_stride10/outputs/avenue_eval10_volume_lejepa_scores.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/fusion_exemplar_lejepa_stride10/outputs/avenue_eval10_exemplar_lejepa_fused_scores.csv"),
    )
    parser.add_argument("--exemplar-weight", type=float, default=0.75)
    parser.add_argument("--clip-high", type=float, default=10.0)
    parser.add_argument("--no-floor-zero", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.exemplar_weight <= 1.0:
        raise ValueError("--exemplar-weight must be between 0 and 1")

    exemplar = load_scores(args.exemplar_scores, "exemplar")
    lejepa = load_scores(args.lejepa_scores, "lejepa")

    exemplar_small = exemplar[
        [
            *METADATA_COLUMNS,
            "anomaly_score",
            "distance_app",
            "distance_ang",
            "distance_speed",
            "distance_bkg",
            "nearest_exemplar_volume_id",
        ]
    ].rename(columns={"anomaly_score": "exemplar_score_raw"})
    lejepa_small = lejepa[["volume_id", "anomaly_score"]].rename(columns={"anomaly_score": "lejepa_score_raw"})
    merged = exemplar_small.merge(lejepa_small, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No matching volume_id rows between exemplar and LEJEPA scores.")

    exemplar_norm, exemplar_stats = robust_scale(
        merged["exemplar_score_raw"],
        floor_zero=not args.no_floor_zero,
        clip_high=args.clip_high,
    )
    lejepa_norm, lejepa_stats = robust_scale(
        merged["lejepa_score_raw"],
        floor_zero=not args.no_floor_zero,
        clip_high=args.clip_high,
    )
    lejepa_weight = 1.0 - args.exemplar_weight
    merged["exemplar_score_norm"] = exemplar_norm
    merged["lejepa_score_norm"] = lejepa_norm
    merged["fusion_exemplar_weight"] = args.exemplar_weight
    merged["fusion_lejepa_weight"] = lejepa_weight
    merged["anomaly_score"] = args.exemplar_weight * exemplar_norm + lejepa_weight * lejepa_norm

    # Keep the original exemplar distance breakdown for diagnostics. The fused
    # anomaly score is the canonical score consumed by downstream evaluation.
    merged["video_id"] = merged["video_id"].map(lambda value: f"{int(value):02d}")
    out_df = merged[OUTPUT_COLUMNS].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = {
        "exemplar_scores": str(args.exemplar_scores),
        "lejepa_scores": str(args.lejepa_scores),
        "out": str(args.out),
        "rows": int(len(out_df)),
        "exemplar_weight": float(args.exemplar_weight),
        "lejepa_weight": float(lejepa_weight),
        "normalization": {
            "method": "robust_median_q95",
            "exemplar": exemplar_stats,
            "lejepa": lejepa_stats,
        },
        "raw_rows": {
            "exemplar": int(len(exemplar)),
            "lejepa": int(len(lejepa)),
            "matched": int(len(out_df)),
        },
        "fused_score": {
            "min": float(out_df["anomaly_score"].min()),
            "mean": float(out_df["anomaly_score"].mean()),
            "median": float(out_df["anomaly_score"].median()),
            "p95": float(out_df["anomaly_score"].quantile(0.95)),
            "max": float(out_df["anomaly_score"].max()),
        },
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(out_df)} fused score rows -> {args.out}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
