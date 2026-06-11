"""Analyze train/test feature distribution differences.

This script is an EDA step, not a training step. It compares the normal
training split against the test split for:

- appearance embeddings
- motion angle histogram
- motion speed vector
- background/stationary attributes
- per-region distribution shifts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_PREFIX = "appearance_resnet18_emb_"
ANG_PREFIX = "motion_angle_hist_"
SPD_PREFIX = "motion_speed_"
MOTION_SCALARS = [
    "motion_background_cls",
    "motion_stationary_fraction",
    "motion_moving_fraction",
    "motion_mean_magnitude",
    "motion_max_magnitude",
]


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "appearance": sorted([c for c in df.columns if c.startswith(APP_PREFIX)]),
        "motion_angle": sorted([c for c in df.columns if c.startswith(ANG_PREFIX)]),
        "motion_speed": sorted([c for c in df.columns if c.startswith(SPD_PREFIX)]),
        "motion_scalar": [c for c in MOTION_SCALARS if c in df.columns],
    }
    missing = [name for name, cols in groups.items() if not cols]
    if missing:
        raise RuntimeError(f"Missing feature groups in input table: {missing}")
    return groups


def safe_std(x: pd.Series) -> float:
    value = float(x.std(ddof=0))
    return value if value > 1e-12 else 1e-12


def feature_stats(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        train_mean = float(train[col].mean())
        test_mean = float(test[col].mean())
        train_std = safe_std(train[col])
        test_std = safe_std(test[col])
        pooled = float(np.sqrt((train_std * train_std + test_std * test_std) / 2.0))
        rows.append(
            {
                "group": group,
                "feature": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "mean_diff_test_minus_train": test_mean - train_mean,
                "abs_mean_diff": abs(test_mean - train_mean),
                "cohen_d": (test_mean - train_mean) / pooled if pooled > 0 else 0.0,
                "abs_cohen_d": abs((test_mean - train_mean) / pooled) if pooled > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def vector_norm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float32)
    return np.linalg.norm(x, axis=1)


def summarize_vector_group(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    train_norm = vector_norm(train, cols)
    test_norm = vector_norm(test, cols)
    train_mean_vec = train[cols].to_numpy(dtype=np.float32).mean(axis=0)
    test_mean_vec = test[cols].to_numpy(dtype=np.float32).mean(axis=0)
    return {
        "dims": len(cols),
        "train_norm_mean": float(train_norm.mean()),
        "test_norm_mean": float(test_norm.mean()),
        "train_norm_std": float(train_norm.std()),
        "test_norm_std": float(test_norm.std()),
        "mean_vector_l2_shift": float(np.linalg.norm(test_mean_vec - train_mean_vec)),
    }


def split_summary(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict:
    train = df[df["split"] == "training"]
    test = df[df["split"] == "testing"]
    out = {
        "rows_total": int(len(df)),
        "rows_by_split": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "regions": int(df["region_id"].nunique()),
        "train_videos": int(train["video_id"].nunique()),
        "test_videos": int(test["video_id"].nunique()),
        "appearance": summarize_vector_group(train, test, groups["appearance"]),
        "motion_angle": summarize_vector_group(train, test, groups["motion_angle"]),
        "motion_speed": summarize_vector_group(train, test, groups["motion_speed"]),
    }
    for col in groups["motion_scalar"]:
        out[col] = {
            "train_mean": float(train[col].mean()),
            "test_mean": float(test[col].mean()),
            "train_std": float(train[col].std(ddof=0)),
            "test_std": float(test[col].std(ddof=0)),
        }
    return out


def region_shift_summary(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for region_id, region_df in df.groupby("region_id"):
        train = region_df[region_df["split"] == "training"]
        test = region_df[region_df["split"] == "testing"]
        if train.empty or test.empty:
            continue
        row = {"region_id": int(region_id), "train_rows": int(len(train)), "test_rows": int(len(test))}
        for group_name in ["appearance", "motion_angle", "motion_speed"]:
            cols = groups[group_name]
            train_mean = train[cols].to_numpy(dtype=np.float32).mean(axis=0)
            test_mean = test[cols].to_numpy(dtype=np.float32).mean(axis=0)
            row[f"{group_name}_mean_l2_shift"] = float(np.linalg.norm(test_mean - train_mean))
        row["stationary_train_mean"] = float(train["motion_stationary_fraction"].mean())
        row["stationary_test_mean"] = float(test["motion_stationary_fraction"].mean())
        row["moving_train_mean"] = float(train["motion_moving_fraction"].mean())
        row["moving_test_mean"] = float(test["motion_moving_fraction"].mean())
        row["background_train_rate"] = float(train["motion_background_cls"].mean())
        row["background_test_rate"] = float(test["motion_background_cls"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("appearance_mean_l2_shift", ascending=False)


def video_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["split", "video_id"])
    rows = []
    for (split, video_id), part in grouped:
        rows.append(
            {
                "split": split,
                "video_id": video_id,
                "rows": int(len(part)),
                "background_rate": float(part["motion_background_cls"].mean()),
                "stationary_mean": float(part["motion_stationary_fraction"].mean()),
                "moving_mean": float(part["motion_moving_fraction"].mean()),
                "mean_magnitude": float(part["motion_mean_magnitude"].mean()),
                "max_magnitude_mean": float(part["motion_max_magnitude"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "video_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/feature_distribution_analysis"))
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    if not args.features.exists() or args.features.stat().st_size == 0:
        raise RuntimeError(f"Feature table is missing or empty: {args.features}")

    df = pd.read_csv(args.features)
    if set(df["split"].unique()) != {"training", "testing"}:
        raise RuntimeError(f"Expected training/testing splits, got: {sorted(df['split'].unique())}")

    groups = feature_groups(df)
    train = df[df["split"] == "training"]
    test = df[df["split"] == "testing"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = split_summary(df, groups)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    stats_frames = []
    for group_name, cols in groups.items():
        stats_frames.append(feature_stats(train, test, cols, group_name))
    all_stats = pd.concat(stats_frames, ignore_index=True)
    all_stats.to_csv(args.out_dir / "feature_train_test_stats.csv", index=False)

    top_shift = all_stats.sort_values("abs_cohen_d", ascending=False).head(args.top_k)
    top_shift.to_csv(args.out_dir / "top_shifted_features.csv", index=False)

    appearance_stats = all_stats[all_stats["group"] == "appearance"].sort_values("abs_cohen_d", ascending=False)
    motion_stats = all_stats[all_stats["group"] != "appearance"].sort_values("abs_cohen_d", ascending=False)
    appearance_stats.head(args.top_k).to_csv(args.out_dir / "appearance_top_shifted_dims.csv", index=False)
    motion_stats.head(args.top_k).to_csv(args.out_dir / "motion_top_shifted_features.csv", index=False)

    region_shift_summary(df, groups).to_csv(args.out_dir / "region_shift_summary.csv", index=False)
    video_summary(df).to_csv(args.out_dir / "video_motion_summary.csv", index=False)

    print(f"Wrote feature distribution analysis -> {args.out_dir}")
    print(f"Rows: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print("Key summary:")
    print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
