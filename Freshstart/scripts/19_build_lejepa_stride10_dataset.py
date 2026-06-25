"""Build the feature-level LEJEPA dataset for Avenue stride-10 volumes.

This prepares a normal-only self-supervised dataset from the existing
stride-10 model features. It does not use anomaly labels.

Outputs:
- train/test feature arrays as compressed .npz
- train/test metadata CSVs
- train-only normalization statistics
- feature schema JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_PREFIX = "appearance_resnet18_emb_"
ANGLE_PREFIX = "motion_angle_hist_"
SPEED_PREFIX = "motion_speed_"
BKG_COLUMNS = ["motion_stationary_fraction", "motion_background_cls"]
META_COLUMNS = ["volume_id", "split", "video_id", "start_frame", "end_frame", "region_id", "x", "y", "w", "h"]


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    app = sorted([col for col in df.columns if col.startswith(APP_PREFIX)])
    angle = sorted([col for col in df.columns if col.startswith(ANGLE_PREFIX)])
    speed = sorted([col for col in df.columns if col.startswith(SPEED_PREFIX)])
    missing = []
    if not app:
        missing.append(APP_PREFIX)
    if len(angle) != 12:
        missing.append("12 motion_angle_hist columns")
    if len(speed) != 12:
        missing.append("12 motion_speed columns")
    for col in BKG_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing}")
    return {
        "appearance": app,
        "angle": angle,
        "speed": speed,
        "background": BKG_COLUMNS,
        "all": app + angle + speed + BKG_COLUMNS,
    }


def standardize(train_x: np.ndarray, test_x: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, eps)
    train_z = ((train_x - mean) / std).astype(np.float32)
    test_z = ((test_x - mean) / std).astype(np.float32)
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": eps,
        "normalization": "zscore_from_training_split_only",
    }
    return train_z, test_z, stats


def write_metadata(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[META_COLUMNS].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/lejepa_stride10/datasets"))
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    if df.empty:
        raise RuntimeError(f"No rows found in {args.features}")

    cols = feature_columns(df)
    train = df[df["split"] == "training"].copy()
    test = df[df["split"] == "testing"].copy()
    if train.empty or test.empty:
        raise RuntimeError("Expected both training and testing rows in the feature table.")

    train_x = train[cols["all"]].to_numpy(dtype=np.float32)
    test_x = test[cols["all"]].to_numpy(dtype=np.float32)
    train_z, test_z, norm_stats = standardize(train_x, test_x, args.eps)

    np.savez_compressed(
        args.out_dir / "avenue_eval10_lejepa_train_features.npz",
        features=train_z,
        region_id=train["region_id"].to_numpy(dtype=np.int32),
        video_id=train["video_id"].astype(str).to_numpy(),
        start_frame=train["start_frame"].to_numpy(dtype=np.int32),
        end_frame=train["end_frame"].to_numpy(dtype=np.int32),
        volume_id=train["volume_id"].astype(str).to_numpy(),
    )
    np.savez_compressed(
        args.out_dir / "avenue_eval10_lejepa_test_features.npz",
        features=test_z,
        region_id=test["region_id"].to_numpy(dtype=np.int32),
        video_id=test["video_id"].astype(str).to_numpy(),
        start_frame=test["start_frame"].to_numpy(dtype=np.int32),
        end_frame=test["end_frame"].to_numpy(dtype=np.int32),
        volume_id=test["volume_id"].astype(str).to_numpy(),
    )

    write_metadata(args.out_dir / "avenue_eval10_lejepa_train_metadata.csv", train)
    write_metadata(args.out_dir / "avenue_eval10_lejepa_test_metadata.csv", test)

    schema = {
        "source_features": str(args.features),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_dim": int(len(cols["all"])),
        "feature_groups": {key: value for key, value in cols.items() if key != "all"},
        "feature_columns": cols["all"],
        "normalization": norm_stats,
        "notes": [
            "Training split is normal-only Avenue data.",
            "No avenue.mat or spatial GT labels are used in this dataset.",
            "This is the first feature-level LEJEPA dataset; later scripts can build temporal/context pairs from these arrays.",
        ],
    }
    (args.out_dir / "avenue_eval10_lejepa_feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote train features -> {args.out_dir / 'avenue_eval10_lejepa_train_features.npz'} {train_z.shape}")
    print(f"Wrote test features -> {args.out_dir / 'avenue_eval10_lejepa_test_features.npz'} {test_z.shape}")
    print(f"Wrote metadata/schema -> {args.out_dir}")


if __name__ == "__main__":
    main()
