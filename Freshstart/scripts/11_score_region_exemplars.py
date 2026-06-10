"""Score volumes with a trained region-specific exemplar model."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


CLS_COLUMN = "motion_background_cls"


def zero_motion_when_background(df: pd.DataFrame, comps: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    background = out[CLS_COLUMN].to_numpy(dtype=np.int32) == 1
    for group in ["ang", "speed", "bkg"]:
        out.loc[background, comps[group]] = 0.0
    return out


def row_components(row: pd.Series, comps: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        "app": row[comps["app"]].to_numpy(dtype=np.float32),
        "ang": row[comps["ang"]].to_numpy(dtype=np.float32),
        "speed": row[comps["speed"]].to_numpy(dtype=np.float32),
        "bkg": row[comps["bkg"]].to_numpy(dtype=np.float32),
    }


def component_distances(candidate: dict[str, np.ndarray], exemplars: dict, normalizers: dict[str, float]) -> dict[str, np.ndarray]:
    d_app = np.linalg.norm(exemplars["app"] - candidate["app"], axis=1) / normalizers["app"]
    d_ang = np.linalg.norm(exemplars["ang"] - candidate["ang"], axis=1) / normalizers["ang"]
    d_speed = np.linalg.norm(exemplars["speed"] - candidate["speed"], axis=1) / normalizers["speed"]
    d_bkg = np.linalg.norm(exemplars["bkg"] - candidate["bkg"], axis=1) / normalizers["bkg"]
    total = d_app + d_ang + d_speed + d_bkg
    return {
        "total": total,
        "app": d_app,
        "ang": d_ang,
        "speed": d_speed,
        "bkg": d_bkg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/avenue_eval10_region_exemplars.pkl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument("--split", choices=["testing", "training", "all"], default="testing")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.model.open("rb") as handle:
        model = pickle.load(handle)

    comps = model["component_columns"]
    normalizers = model["normalizers"]
    df = pd.read_csv(args.features)
    if args.split != "all":
        df = df[df["split"] == args.split].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    if df.empty:
        raise RuntimeError("No rows selected for scoring.")

    df = zero_motion_when_background(df, comps)
    rows = []
    for idx, row in df.iterrows():
        region_id = int(row["region_id"])
        exemplars = model["regions"].get(region_id)
        if exemplars is None:
            raise RuntimeError(f"No exemplars found for region_id={region_id}")

        candidate = row_components(row, comps)
        distances = component_distances(candidate, exemplars, normalizers)
        best_idx = int(np.argmin(distances["total"]))

        rows.append(
            {
                "volume_id": row["volume_id"],
                "split": row["split"],
                "video_id": row["video_id"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "region_id": region_id,
                "x": int(row["x"]),
                "y": int(row["y"]),
                "w": int(row["w"]),
                "h": int(row["h"]),
                "anomaly_score": float(distances["total"][best_idx]),
                "distance_app": float(distances["app"][best_idx]),
                "distance_ang": float(distances["ang"][best_idx]),
                "distance_speed": float(distances["speed"][best_idx]),
                "distance_bkg": float(distances["bkg"][best_idx]),
                "nearest_exemplar_volume_id": exemplars["volume_ids"][best_idx],
            }
        )
        if len(rows) % 1000 == 0:
            print(f"Scored {len(rows)} rows...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} score rows -> {args.out}")


if __name__ == "__main__":
    main()
