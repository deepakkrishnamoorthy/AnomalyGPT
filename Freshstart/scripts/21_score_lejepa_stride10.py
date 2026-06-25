"""Score Avenue stride-10 test volumes with the feature-level LEJEPA model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class LejepaFeatureModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        region_count: int,
        region_emb_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.region_embedding = nn.Embedding(region_count, region_emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + region_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + region_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, x: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        region = self.region_embedding(region_id)
        return self.encoder(torch.cat([x, region], dim=1))

    def forward(self, x_context: torch.Tensor, x_target: torch.Tensor, region_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        region = self.region_embedding(region_id)
        pred = self.predictor(torch.cat([self.encode(x_context, region_id), region], dim=1))
        target = self.encode(x_target, region_id)
        return pred, target


def load_model(path: Path, device: torch.device) -> tuple[LejepaFeatureModel, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = LejepaFeatureModel(
        feature_dim=int(config["feature_dim"]),
        region_count=int(config["region_count"]),
        region_emb_dim=int(config["region_emb_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def masked_context(features: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Deterministic context mask for scoring.

    During training the mask is random. During scoring we use a stable evenly
    spaced mask so every run produces the same anomaly scores.
    """

    if mask_ratio <= 0:
        return features
    dim = features.shape[1]
    mask_count = max(1, int(round(dim * mask_ratio)))
    idx = torch.linspace(0, dim - 1, mask_count, device=features.device).round().long().unique()
    context = features.clone()
    context[:, idx] = 0.0
    return context


def component_columns(feature_dim: int) -> dict[str, slice]:
    # Schema from 19_build_lejepa_stride10_dataset.py:
    # 512 appearance + 12 angle + 12 speed + 2 background = 538
    if feature_dim < 538:
        raise ValueError(f"Expected feature_dim >= 538, got {feature_dim}")
    return {
        "app": slice(0, 512),
        "ang": slice(512, 524),
        "speed": slice(524, 536),
        "bkg": slice(536, 538),
    }


def mse_by_slice(a: torch.Tensor, b: torch.Tensor, s: slice) -> torch.Tensor:
    return torch.mean((a[:, s] - b[:, s]) ** 2, dim=1)


def score_batches(
    model: LejepaFeatureModel,
    features: np.ndarray,
    region_ids: np.ndarray,
    *,
    batch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    feature_dim = features.shape[1]
    slices = component_columns(feature_dim)
    scores = {"total": [], "app": [], "ang": [], "speed": [], "bkg": []}
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(len(features), start + batch_size)
            x = torch.from_numpy(features[start:end].astype(np.float32)).to(device)
            region = torch.from_numpy(region_ids[start:end].astype(np.int64)).to(device)
            context = masked_context(x, mask_ratio)
            pred, target = model(context, x, region)
            # Latent total score.
            total = torch.mean((pred - target) ** 2, dim=1)
            scores["total"].append(total.cpu().numpy())

            # Component scores are computed by re-encoding component-masked inputs.
            # This is approximate but keeps explanations aligned with our existing
            # appearance/motion/background dashboard schema.
            for name, s in slices.items():
                x_component = torch.zeros_like(x)
                x_component[:, s] = x[:, s]
                c_component = masked_context(x_component, mask_ratio)
                pred_c, target_c = model(c_component, x_component, region)
                scores[name].append(torch.mean((pred_c - target_c) ** 2, dim=1).cpu().numpy())

            if end % 10000 == 0:
                print(f"Scored {end} volumes...")

    return {key: np.concatenate(values).astype(np.float32) for key, values in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("experiments/lejepa_stride10/models/avenue_eval10_lejepa_feature_model.pt"))
    parser.add_argument("--test-features", type=Path, default=Path("experiments/lejepa_stride10/datasets/avenue_eval10_lejepa_test_features.npz"))
    parser.add_argument("--test-metadata", type=Path, default=Path("experiments/lejepa_stride10/datasets/avenue_eval10_lejepa_test_metadata.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/lejepa_stride10/outputs/avenue_eval10_lejepa_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Defaults to the training mask ratio from the checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.model, device)
    mask_ratio = float(config["mask_ratio"] if args.mask_ratio is None else args.mask_ratio)
    data = np.load(args.test_features, allow_pickle=True)
    features = data["features"].astype(np.float32)
    region_ids = data["region_id"].astype(np.int64)
    metadata = pd.read_csv(args.test_metadata)
    if len(metadata) != len(features):
        raise RuntimeError(f"Metadata/features length mismatch: {len(metadata)} vs {len(features)}")

    scores = score_batches(
        model,
        features,
        region_ids,
        batch_size=args.batch_size,
        mask_ratio=mask_ratio,
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
            "anomaly_score",
            "distance_app",
            "distance_ang",
            "distance_speed",
            "distance_bkg",
            "nearest_exemplar_volume_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in metadata.iterrows():
            writer.writerow(
                {
                    "volume_id": row["volume_id"],
                    "split": row["split"],
                    "video_id": row["video_id"],
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "region_id": int(row["region_id"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "w": int(row["w"]),
                    "h": int(row["h"]),
                    "anomaly_score": float(scores["total"][idx]),
                    "distance_app": float(scores["app"][idx]),
                    "distance_ang": float(scores["ang"][idx]),
                    "distance_speed": float(scores["speed"][idx]),
                    "distance_bkg": float(scores["bkg"][idx]),
                    "nearest_exemplar_volume_id": "latent_model",
                }
            )
    print(f"Wrote {len(metadata)} LEJEPA score rows -> {args.out}")


if __name__ == "__main__":
    main()
