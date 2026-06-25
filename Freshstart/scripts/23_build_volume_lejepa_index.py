"""Build Route B-lite volume-level LEJEPA index for stride-10 Avenue volumes.

The index is a lightweight CSV over the existing manifest. It does not copy
frames or volumes. Training/scoring scripts will read frame crops lazily from
data/avenue_frames using these rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEEP_COLUMNS = [
    "volume_id",
    "split",
    "video_id",
    "start_frame",
    "end_frame",
    "depth",
    "region_id",
    "x",
    "y",
    "w",
    "h",
    "source_width",
    "source_height",
]


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/lejepa_volume_stride10/datasets"))
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    args = parser.parse_args()

    rows = []
    for row in iter_manifest(args.manifest):
        rows.append({key: row[key] for key in KEEP_COLUMNS})
        if args.limit is not None and len(rows) >= args.limit:
            break
    if not rows:
        raise SystemExit("No rows selected from manifest.")

    df = pd.DataFrame(rows)
    train = df[df["split"] == "training"].copy()
    test = df[df["split"] == "testing"].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.out_dir / "avenue_eval10_volume_lejepa_index.csv"
    train_path = args.out_dir / "avenue_eval10_volume_lejepa_train_index.csv"
    test_path = args.out_dir / "avenue_eval10_volume_lejepa_test_index.csv"
    summary_path = args.out_dir / "avenue_eval10_volume_lejepa_index_summary.json"

    df.to_csv(all_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    summary = {
        "source_manifest": str(args.manifest),
        "rows_total": int(len(df)),
        "rows_training": int(len(train)),
        "rows_testing": int(len(test)),
        "depth_values": sorted(int(v) for v in df["depth"].unique().tolist()),
        "regions": int(df["region_id"].nunique()),
        "notes": [
            "This index references existing frames in data/avenue_frames.",
            "It does not copy or materialize volume tensors.",
            "Ground truth is not used for training.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote all index -> {all_path} rows={len(df)}")
    print(f"Wrote train index -> {train_path} rows={len(train)}")
    print(f"Wrote test index -> {test_path} rows={len(test)}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
