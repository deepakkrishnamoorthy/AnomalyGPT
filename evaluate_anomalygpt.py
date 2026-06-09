"""Evaluation entry point for Physics-Grounded AnomalyGPT."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/anomalygpt"))
    args = parser.parse_args()
    raise NotImplementedError(
        f"Evaluation wiring is pending model integration. Checkpoint: {args.checkpoint}; "
        f"dataset: {args.dataset_root}"
    )


if __name__ == "__main__":
    main()
