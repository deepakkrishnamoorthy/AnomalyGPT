"""Training entry point for Physics-Grounded AnomalyGPT."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/anomalygpt/default.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/anomalygpt"))
    args = parser.parse_args()
    raise NotImplementedError(
        f"Training wiring is pending WalkGPT integration. Config: {args.config}; "
        f"dataset: {args.dataset_root}"
    )


if __name__ == "__main__":
    main()
