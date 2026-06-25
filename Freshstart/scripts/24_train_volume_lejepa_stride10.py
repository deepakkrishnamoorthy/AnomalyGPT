"""Train Route B-lite volume-level LEJEPA on stride-10 Avenue volumes.

Normal-only task:

    grayscale frames 1-5 -> predict latent of grayscale frames 6-10

This is intentionally small and practical: it lazily reads crops from
data/avenue_frames instead of materializing another huge volume dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frame_dir(frames_root: Path, split: str, video_id: str | int) -> Path:
    return frames_root / SPLIT_TO_FRAME_DIR[split] / f"{int(video_id):02d}"


def read_gray_crop(path: Path, row: pd.Series, image_size: int) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError(f"Could not read frame {path}")
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pad_bottom = max(0, y + h - frame.shape[0])
    pad_right = max(0, x + w - frame.shape[1])
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
    crop = frame[y : y + h, x : x + w]
    if crop.shape[0] != image_size or crop.shape[1] != image_size:
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0


class AvenueVolumeDataset(Dataset):
    def __init__(self, index_csv: Path, frames_root: Path, image_size: int, limit: int | None = None):
        self.df = pd.read_csv(index_csv)
        if limit is not None:
            self.df = self.df.head(limit).copy()
        self.frames_root = frames_root
        self.image_size = image_size

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vdir = frame_dir(self.frames_root, row["split"], row["video_id"])
        frames = []
        for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            frames.append(read_gray_crop(vdir / f"{frame_idx:05d}.jpg", row, self.image_size))
        volume = np.stack(frames, axis=0)
        if volume.shape[0] != 10:
            raise RuntimeError(f"Expected 10 frames for {row['volume_id']}, got {volume.shape[0]}")
        context = torch.from_numpy(volume[:5][None, ...])
        target = torch.from_numpy(volume[5:][None, ...])
        region_id = torch.tensor(int(row["region_id"]), dtype=torch.long)
        return context, target, region_id


class VolumeEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VolumeLejepa(nn.Module):
    def __init__(self, latent_dim: int, region_count: int, region_emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.context_encoder = VolumeEncoder(latent_dim)
        self.target_encoder = VolumeEncoder(latent_dim)
        self.region_embedding = nn.Embedding(region_count, region_emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + region_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, context: torch.Tensor, target: torch.Tensor, region_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        region = self.region_embedding(region_id)
        context_latent = self.context_encoder(context)
        pred = self.predictor(torch.cat([context_latent, region], dim=1))
        with torch.no_grad():
            target_latent = self.target_encoder(target)
        return pred, target_latent


def update_target_encoder(model: VolumeLejepa, momentum: float) -> None:
    with torch.no_grad():
        for target_param, context_param in zip(model.target_encoder.parameters(), model.context_encoder.parameters()):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)


def run_epoch(
    model: VolumeLejepa,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    momentum: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    loss_fn = nn.MSELoss()
    for context, target, region_id in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        region_id = region_id.to(device, non_blocking=True)
        pred, target_latent = model(context, target, region_id)
        loss = loss_fn(pred, target_latent)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_target_encoder(model, momentum)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, default=Path("experiments/lejepa_volume_stride10/datasets/avenue_eval10_volume_lejepa_train_index.csv"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out", type=Path, default=Path("experiments/lejepa_volume_stride10/models/avenue_eval10_volume_lejepa.pt"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--region-emb-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--target-momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = AvenueVolumeDataset(args.train_index, args.frames_root, args.image_size, limit=args.limit)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 training rows.")
    region_count = int(dataset.df["region_id"].max()) + 1

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VolumeLejepa(
        latent_dim=args.latent_dim,
        region_count=region_count,
        region_emb_dim=args.region_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.target_encoder.load_state_dict(model.context_encoder.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.target_momentum)
        val_loss = run_epoch(model, val_loader, None, device, args.target_momentum)
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
            "latent_dim": args.latent_dim,
            "region_count": region_count,
            "region_emb_dim": args.region_emb_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "image_size": args.image_size,
            "target_momentum": args.target_momentum,
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
                "train_index": str(args.train_index),
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
