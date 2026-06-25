"""Train a feature-level LEJEPA-style latent predictor on stride-10 Avenue features.

This is a lightweight normal-only self-supervised baseline:

- input: normalized feature vector F for a 10-frame region volume
- encoder: maps F to latent z
- predictor: predicts selected target latent dimensions from masked/context dimensions
- objective: MSE in latent space

The model is trained only on Avenue training volumes. No anomaly labels are used.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class FeatureDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.features = torch.from_numpy(data["features"].astype(np.float32))
        self.region_id = torch.from_numpy(data["region_id"].astype(np.int64))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.region_id[idx]


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
        with torch.no_grad():
            target = self.encode(x_target, region_id)
        return pred, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mask(batch: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(batch) < mask_ratio
    if mask_ratio <= 0:
        mask = torch.zeros_like(batch, dtype=torch.bool)
    context = batch.clone()
    context[mask] = 0.0
    return context, batch


def run_epoch(
    model: LejepaFeatureModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mask_ratio: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    loss_fn = nn.MSELoss()
    for features, region_id in loader:
        features = features.to(device)
        region_id = region_id.to(device)
        context, target_input = make_mask(features, mask_ratio)
        pred, target = model(context, target_input, region_id)
        loss = loss_fn(pred, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, default=Path("experiments/lejepa_stride10/datasets/avenue_eval10_lejepa_train_features.npz"))
    parser.add_argument("--out", type=Path, default=Path("experiments/lejepa_stride10/models/avenue_eval10_lejepa_feature_model.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--region-emb-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = FeatureDataset(args.train_features)
    feature_dim = int(dataset.features.shape[1])
    region_count = int(dataset.region_id.max().item()) + 1

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = LejepaFeatureModel(
        feature_dim=feature_dim,
        region_count=region_count,
        region_emb_dim=args.region_emb_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.mask_ratio)
        val_loss = run_epoch(model, val_loader, None, device, args.mask_ratio)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "feature_dim": feature_dim,
            "region_count": region_count,
            "region_emb_dim": args.region_emb_dim,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "mask_ratio": args.mask_ratio,
            "seed": args.seed,
        },
        "history": history,
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, args.out)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "model_path": str(args.out),
                "train_features": str(args.train_features),
                "train_rows": train_size,
                "val_rows": val_size,
                "best_val_loss": best_val,
                "config": checkpoint["config"],
                "history": history,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote model -> {args.out}")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
