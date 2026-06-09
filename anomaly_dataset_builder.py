"""Build AnomalyGPT dataset records from raw VAD datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_dataset(raw_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Dataset conversion will be implemented after UCSD/Avenue downloads finish."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/anomalygpt"))
    args = parser.parse_args()
    build_dataset(args.raw_root, args.output_root)


if __name__ == "__main__":
    main()
