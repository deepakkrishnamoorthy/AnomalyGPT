"""Score stride-10 Avenue volumes with the Route B-lite volume LEJEPA model.

The model was trained only on normal training volumes to predict the latent
representation of frames 6-10 from frames 1-5. At test time, the latent
prediction error becomes the anomaly score for each spatio-temporal volume.

The output CSV intentionally matches the metadata schema used by
11_score_region_exemplars.py so the existing frame projection and evaluation
scripts can be reused directly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}

FIELDNAMES = [
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


class AvenueVolumeScoreDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        frames_root: Path,
        image_size: int,
        *,
        start_row: int = 1,
        limit: int | None = None,
    ):
        df = pd.read_csv(index_csv)
        if start_row < 1:
            raise ValueError("--start-row must be >= 1")
        df = df.iloc[start_row - 1 :].copy()
        if limit is not None:
            df = df.head(limit).copy()
        if df.empty:
            raise RuntimeError("No rows selected for scoring.")
        self.df = df.reset_index(drop=True)
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
        return context, target, region_id, torch.tensor(idx, dtype=torch.long)


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
        target_latent = self.target_encoder(target)
        return pred, target_latent


def load_model(model_path: Path, device: torch.device) -> tuple[VolumeLejepa, dict]:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = VolumeLejepa(
        latent_dim=int(config["latent_dim"]),
        region_count=int(config["region_count"]),
        region_emb_dim=int(config["region_emb_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def output_row(row: pd.Series, score: float, pred_norm: float, target_norm: float) -> dict:
    # The LEJEPA scorer has one latent prediction-error signal, not separate
    # EVAL-style appearance/direction/speed/background distances.
    return {
        "volume_id": row["volume_id"],
        "split": row["split"],
        "video_id": f"{int(row['video_id']):02d}",
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        "region_id": int(row["region_id"]),
        "x": int(row["x"]),
        "y": int(row["y"]),
        "w": int(row["w"]),
        "h": int(row["h"]),
        "anomaly_score": float(score),
        "distance_app": float(score),
        "distance_ang": 0.0,
        "distance_speed": 0.0,
        "distance_bkg": 0.0,
        "nearest_exemplar_volume_id": f"volume_lejepa_pred_norm={pred_norm:.6f};target_norm={target_norm:.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("experiments/lejepa_volume_stride10/models/avenue_eval10_volume_lejepa.pt"))
    parser.add_argument("--test-index", type=Path, default=Path("experiments/lejepa_volume_stride10/datasets/avenue_eval10_volume_lejepa_test_index.csv"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out", type=Path, default=Path("experiments/lejepa_volume_stride10/outputs/avenue_eval10_volume_lejepa_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--start-row", type=int, default=1, help="1-based selected test-index row to start scoring from.")
    parser.add_argument("--append", action="store_true", help="Append scores to an existing CSV instead of rewriting it.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.model, device)
    image_size = int(config.get("image_size", 128))
    dataset = AvenueVolumeScoreDataset(
        args.test_index,
        args.frames_root,
        image_size,
        start_row=args.start_row,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not args.append or not args.out.exists() or args.out.stat().st_size == 0
    written = 0

    with args.out.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with torch.inference_mode():
            for context, target, region_id, local_idx in loader:
                context = context.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                region_id = region_id.to(device, non_blocking=True)
                pred, target_latent = model(context, target, region_id)
                scores = torch.mean((pred - target_latent) ** 2, dim=1)
                pred_norms = torch.linalg.norm(pred, dim=1)
                target_norms = torch.linalg.norm(target_latent, dim=1)

                for idx, score, pred_norm, target_norm in zip(
                    local_idx.cpu().numpy(),
                    scores.cpu().numpy(),
                    pred_norms.cpu().numpy(),
                    target_norms.cpu().numpy(),
                ):
                    row = dataset.df.iloc[int(idx)]
                    writer.writerow(output_row(row, float(score), float(pred_norm), float(target_norm)))
                    written += 1

                if written and written % 1000 == 0:
                    print(f"Scored {written} rows from start row {args.start_row}...")

    print(f"Wrote {written} LEJEPA score rows from start row {args.start_row} -> {args.out}")


if __name__ == "__main__":
    main()
