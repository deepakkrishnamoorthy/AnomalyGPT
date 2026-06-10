"""Join appearance and motion features into one model-ready table.

The output keeps component boundaries explicit so EVAL-style distance functions
can treat appearance, angle, speed, stationary/background, and class terms
separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
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


def require_nonempty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"{path} is empty. Wait for feature extraction to finish, then rerun.")


def read_csv(path: Path, name: str) -> pd.DataFrame:
    require_nonempty(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{name} has no rows: {path}")
    if df["volume_id"].duplicated().any():
        dupes = int(df["volume_id"].duplicated().sum())
        raise RuntimeError(f"{name} has duplicate volume_id rows: {dupes}")
    return df


def validate_keys(left: pd.DataFrame, right: pd.DataFrame, right_name: str) -> None:
    missing = set(left["volume_id"]) - set(right["volume_id"])
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"{right_name} is missing {len(missing)} volume_ids. Sample: {sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appearance", type=Path, default=Path("features/avenue_eval10_appearance_resnet18.csv"))
    parser.add_argument("--motion", type=Path, default=Path("features/avenue_eval10_motion_attributes.csv"))
    parser.add_argument("--yolo", type=Path, default=None, help="Optional YOLO object metadata CSV.")
    parser.add_argument("--out", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    args = parser.parse_args()

    appearance = read_csv(args.appearance, "appearance")
    motion = read_csv(args.motion, "motion")
    validate_keys(appearance, motion, "motion")

    motion_feature_cols = [c for c in motion.columns if c not in KEY_COLUMNS and c != "depth"]
    merged = appearance.merge(motion[["volume_id"] + motion_feature_cols], on="volume_id", how="left", validate="one_to_one")

    if args.yolo is not None:
        yolo = read_csv(args.yolo, "yolo")
        validate_keys(merged, yolo, "yolo")
        yolo_cols = [c for c in yolo.columns if c not in KEY_COLUMNS and c not in merged.columns]
        merged = merged.merge(yolo[["volume_id"] + yolo_cols], on="volume_id", how="left", validate="one_to_one")

    merged["feature_schema"] = "appearance_resnet18_512+eval_motion_12ang_12speed+bkg"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {len(merged)} model feature rows -> {args.out}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Splits: {merged['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
