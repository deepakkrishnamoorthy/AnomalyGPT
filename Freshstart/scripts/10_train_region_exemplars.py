"""Train region-specific normal exemplar sets from model features.

This implements the EVAL-style greedy exemplar selection:

1. Keep the first normal feature vector for a region.
2. Add a later feature vector only if its distance to every existing exemplar
   in the same region is above a threshold.

The model is normal-only: it uses rows where split == training.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


APP_PREFIX = "appearance_resnet18_emb_"
ANG_PREFIX = "motion_angle_hist_"
SPD_PREFIX = "motion_speed_"
BKG_COLUMNS = ["motion_stationary_fraction"]
CLS_COLUMN = "motion_background_cls"


def component_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    app = sorted([c for c in df.columns if c.startswith(APP_PREFIX)])
    ang = sorted([c for c in df.columns if c.startswith(ANG_PREFIX)])
    speed = sorted([c for c in df.columns if c.startswith(SPD_PREFIX)])
    missing = []
    if not app:
        missing.append(APP_PREFIX)
    if len(ang) != 12:
        missing.append("12 motion_angle_hist columns")
    if len(speed) != 12:
        missing.append("12 motion_speed columns")
    for col in BKG_COLUMNS + [CLS_COLUMN]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing}")
    return {"app": app, "ang": ang, "speed": speed, "bkg": BKG_COLUMNS, "cls": [CLS_COLUMN]}


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


def zero_motion_when_background(df: pd.DataFrame, comps: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    background = out[CLS_COLUMN].to_numpy(dtype=np.int32) == 1
    for group in ["ang", "speed", "bkg"]:
        out.loc[background, comps[group]] = 0.0
    return out


def estimate_max_l2(x: np.ndarray, pair_sample_size: int, rng: np.random.Generator) -> float:
    if len(x) <= 1:
        return 1.0
    n_pairs = max(1, pair_sample_size)
    left = rng.integers(0, len(x), size=n_pairs)
    right = rng.integers(0, len(x), size=n_pairs)
    diff = x[left] - x[right]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return max(float(dist.max()), 1e-6)


def compute_normalizers(df: pd.DataFrame, comps: dict[str, list[str]], pair_sample_size: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {
        "app": estimate_max_l2(matrix(df, comps["app"]), pair_sample_size, rng),
        "ang": estimate_max_l2(matrix(df, comps["ang"]), pair_sample_size, rng),
        "speed": estimate_max_l2(matrix(df, comps["speed"]), pair_sample_size, rng),
        "bkg": estimate_max_l2(matrix(df, comps["bkg"]), pair_sample_size, rng),
    }


def component_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray], normalizers: dict[str, float]) -> np.ndarray:
    d_app = np.linalg.norm(b["app"] - a["app"], axis=1) / normalizers["app"]
    d_ang = np.linalg.norm(b["ang"] - a["ang"], axis=1) / normalizers["ang"]
    d_speed = np.linalg.norm(b["speed"] - a["speed"], axis=1) / normalizers["speed"]
    d_bkg = np.linalg.norm(b["bkg"] - a["bkg"], axis=1) / normalizers["bkg"]
    return d_app + d_ang + d_speed + d_bkg


def row_components(row: pd.Series, comps: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        "app": row[comps["app"]].to_numpy(dtype=np.float32),
        "ang": row[comps["ang"]].to_numpy(dtype=np.float32),
        "speed": row[comps["speed"]].to_numpy(dtype=np.float32),
        "bkg": row[comps["bkg"]].to_numpy(dtype=np.float32),
    }


def exemplar_matrix(rows: list[pd.Series], comps: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        "app": np.stack([row[comps["app"]].to_numpy(dtype=np.float32) for row in rows], axis=0),
        "ang": np.stack([row[comps["ang"]].to_numpy(dtype=np.float32) for row in rows], axis=0),
        "speed": np.stack([row[comps["speed"]].to_numpy(dtype=np.float32) for row in rows], axis=0),
        "bkg": np.stack([row[comps["bkg"]].to_numpy(dtype=np.float32) for row in rows], axis=0),
    }


def select_region_exemplars(region_df: pd.DataFrame, comps: dict[str, list[str]], normalizers: dict[str, float], threshold: float) -> list[pd.Series]:
    exemplars: list[pd.Series] = []
    for _, row in region_df.iterrows():
        if not exemplars:
            exemplars.append(row)
            continue
        candidate = row_components(row, comps)
        current = exemplar_matrix(exemplars, comps)
        distances = component_distance(candidate, current, normalizers)
        if float(distances.min()) > threshold:
            exemplars.append(row)
    return exemplars


def serialize_exemplars(exemplars: list[pd.Series], comps: dict[str, list[str]]) -> dict:
    ids = [str(row["volume_id"]) for row in exemplars]
    meta = [
        {
            "volume_id": str(row["volume_id"]),
            "video_id": str(row["video_id"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "region_id": int(row["region_id"]),
        }
        for row in exemplars
    ]
    mats = exemplar_matrix(exemplars, comps)
    mats["volume_ids"] = ids
    mats["metadata"] = meta
    return mats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/avenue_eval10_region_exemplars.pkl"))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--normalizer-pair-sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-regions", type=int, default=None, help="Debug only: train first N region ids.")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    train = df[df["split"] == "training"].copy()
    if train.empty:
        raise RuntimeError("No training rows found.")

    comps = component_columns(train)
    train = zero_motion_when_background(train, comps)
    normalizers = compute_normalizers(train, comps, args.normalizer_pair_sample_size, args.seed)

    region_ids = sorted(train["region_id"].unique().tolist())
    if args.limit_regions is not None:
        region_ids = region_ids[: args.limit_regions]

    model = {
        "version": "freshstart_eval10_region_exemplars_v1",
        "threshold": args.threshold,
        "normalizers": normalizers,
        "component_columns": comps,
        "regions": {},
    }

    for region_id in region_ids:
        region_df = train[train["region_id"] == region_id].sort_values(["video_id", "start_frame"])
        exemplars = select_region_exemplars(region_df, comps, normalizers, args.threshold)
        model["regions"][int(region_id)] = serialize_exemplars(exemplars, comps)
        print(f"region {int(region_id):03d}: train_rows={len(region_df)} exemplars={len(exemplars)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(model, handle)

    summary_path = args.out.with_suffix(".summary.json")
    summary = {
        "model_path": str(args.out),
        "threshold": args.threshold,
        "normalizers": normalizers,
        "regions": {str(k): len(v["volume_ids"]) for k, v in model["regions"].items()},
        "total_exemplars": int(sum(len(v["volume_ids"]) for v in model["regions"].values())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote exemplar model -> {args.out}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
