"""Extract training-free appearance embeddings for EVAL-style volumes.

This script reads the volume manifest, crops the original RGB frames for each
10-frame spatial volume, runs an ImageNet backbone, averages frame embeddings
over time, and writes one feature row per volume_id.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_backbone(name: str, weights_mode: str, device: torch.device) -> tuple[nn.Module, int]:
    name = name.lower()
    if name != "resnet18":
        raise ValueError("Only resnet18 is wired for the first Freshstart appearance baseline.")

    weights = None
    if weights_mode == "imagenet":
        weights = models.ResNet18_Weights.IMAGENET1K_V1

    net = models.resnet18(weights=weights)
    feature_dim = int(net.fc.in_features)
    net.fc = nn.Identity()
    net.eval()
    net.to(device)
    return net, feature_dim


def preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_rgb_crop(frame_path: Path, row: dict) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_path}")

    x = int(row["x"])
    y = int(row["y"])
    w = int(row["w"])
    h = int(row["h"])
    source_h, source_w = frame.shape[:2]

    pad_bottom = max(0, y + h - source_h)
    pad_right = max(0, x + w - source_w)
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            pad_bottom,
            0,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    crop_bgr = frame[y : y + h, x : x + w]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def load_volume_frames(row: dict, frames_root: Path, transform) -> torch.Tensor:
    split_dir = SPLIT_TO_FRAME_DIR[row["split"]]
    video_dir = frames_root / split_dir / row["video_id"]
    tensors = []

    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame_path = video_dir / f"{frame_idx:05d}.jpg"
        crop_rgb = read_rgb_crop(frame_path, row)
        tensors.append(transform(crop_rgb))

    return torch.stack(tensors, dim=0)


def write_header(handle, metadata_fields: list[str], feature_dim: int) -> csv.DictWriter:
    feature_fields = [f"appearance_resnet18_emb_{idx:03d}" for idx in range(feature_dim)]
    writer = csv.DictWriter(handle, fieldnames=metadata_fields + feature_fields)
    writer.writeheader()
    return writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out", type=Path, default=Path("features/avenue_eval10_appearance_resnet18.csv"))
    parser.add_argument("--backbone", choices=["resnet18"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="Number of frames per network batch.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, feature_dim = build_backbone(args.backbone, args.weights, device)
    transform = preprocess()

    metadata_fields = [
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
        "appearance_backbone",
        "appearance_weights",
        "appearance_pooling",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = write_header(handle, metadata_fields, feature_dim)

        for row in iter_manifest(args.manifest):
            if args.split != "all" and row["split"] != args.split:
                continue
            if args.limit is not None and written >= args.limit:
                break

            frames = load_volume_frames(row, args.frames_root, transform)
            frame_embeddings = []
            with torch.no_grad():
                for start in range(0, frames.shape[0], args.batch_size):
                    batch = frames[start : start + args.batch_size].to(device)
                    emb = model(batch).detach().cpu()
                    frame_embeddings.append(emb)

            volume_embedding = torch.cat(frame_embeddings, dim=0).mean(dim=0).numpy().astype(np.float32)

            out_row = {
                "volume_id": row["volume_id"],
                "split": row["split"],
                "video_id": row["video_id"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "region_id": row["region_id"],
                "x": row["x"],
                "y": row["y"],
                "w": row["w"],
                "h": row["h"],
                "appearance_backbone": args.backbone,
                "appearance_weights": args.weights,
                "appearance_pooling": "mean_frames",
            }
            for idx, value in enumerate(volume_embedding):
                out_row[f"appearance_resnet18_emb_{idx:03d}"] = f"{float(value):.7g}"

            writer.writerow(out_row)
            written += 1

            if written % 1000 == 0:
                print(f"Wrote {written} appearance feature rows...")

    print(f"Wrote {written} appearance feature rows -> {args.out}")


if __name__ == "__main__":
    main()
