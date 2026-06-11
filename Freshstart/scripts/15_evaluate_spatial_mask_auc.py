"""Evaluate compact spatial anomaly maps against Avenue pixel masks.

This uses the Avenue `volLabel` masks from the spatial GT demo. It reports:

- frame ROC AUC using max spatial score per frame, matching EVAL's frame logic
- frame AP
- sampled pixel ROC AUC/AP after resizing compact score grids to mask size
- per-video frame metrics

RBDC/TBDC in the EVAL paper are bbox/track criteria. Avenue's files here are
pixel masks, so this script is a spatial-mask evaluation baseline rather than a
literal RBDC/TBDC reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score


def load_masks(path: Path) -> list[np.ndarray]:
    mat = sio.loadmat(path, squeeze_me=False)
    if "volLabel" not in mat:
        raise RuntimeError(f"No volLabel key found in {path}")
    cells = mat["volLabel"].reshape(-1)
    return [(np.asarray(cell) > 0).astype(np.uint8) for cell in cells]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0:
        return None
    return float(average_precision_score(labels, scores))


def sample_pixels(mask: np.ndarray, score_map: np.ndarray, max_pixels: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = mask.reshape(-1)
    scores = score_map.reshape(-1)
    if len(labels) <= max_pixels:
        return labels, scores
    idx = rng.choice(len(labels), size=max_pixels, replace=False)
    return labels[idx], scores[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-map-dir", type=Path, default=Path("outputs/avenue_eval10_spatial_score_maps"))
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("Avenue Dataset/avenue-spatial-GT/ground_truth_demo/testing_label_mask"),
    )
    parser.add_argument("--out-summary", type=Path, default=Path("outputs/avenue_eval10_spatial_mask_auc_summary.json"))
    parser.add_argument("--out-per-video", type=Path, default=Path("outputs/avenue_eval10_spatial_mask_auc_per_video.csv"))
    parser.add_argument("--pixel-sample-per-video", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_frame_labels = []
    all_frame_scores = []
    all_pixel_labels = []
    all_pixel_scores = []
    per_video = []

    for video_idx in range(1, 22):
        video_id = f"{video_idx:02d}"
        score_path = args.spatial_map_dir / f"{video_id}.npy"
        mask_path = args.mask_dir / f"{video_idx}_label.mat"
        if not score_path.exists():
            raise RuntimeError(f"Missing spatial score map: {score_path}")
        if not mask_path.exists():
            raise RuntimeError(f"Missing mask file: {mask_path}")

        spatial_scores = np.load(score_path)
        masks = load_masks(mask_path)
        if len(masks) != spatial_scores.shape[0]:
            raise RuntimeError(f"Frame count mismatch for video {video_id}: masks={len(masks)} scores={spatial_scores.shape[0]}")

        frame_scores = spatial_scores.max(axis=(1, 2))
        frame_labels = np.array([int(mask.any()) for mask in masks], dtype=np.uint8)
        all_frame_labels.append(frame_labels)
        all_frame_scores.append(frame_scores)

        pixel_labels_video = []
        pixel_scores_video = []
        per_frame_sample = max(1, args.pixel_sample_per_video // len(masks))
        for frame_idx, mask in enumerate(masks):
            score_grid = spatial_scores[frame_idx]
            score_map = cv2.resize(score_grid, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            labels, scores = sample_pixels(mask, score_map, per_frame_sample, rng)
            pixel_labels_video.append(labels)
            pixel_scores_video.append(scores)

        pixel_labels = np.concatenate(pixel_labels_video)
        pixel_scores = np.concatenate(pixel_scores_video)
        all_pixel_labels.append(pixel_labels)
        all_pixel_scores.append(pixel_scores)

        per_video.append(
            {
                "video_id": video_id,
                "frames": int(len(masks)),
                "anomaly_frames": int(frame_labels.sum()),
                "frame_roc_auc": safe_auc(frame_labels, frame_scores),
                "frame_average_precision": safe_ap(frame_labels, frame_scores),
                "sampled_pixels": int(len(pixel_labels)),
                "positive_pixels_sampled": int(pixel_labels.sum()),
                "pixel_roc_auc_sampled": safe_auc(pixel_labels, pixel_scores),
                "pixel_average_precision_sampled": safe_ap(pixel_labels, pixel_scores),
                "score_min": float(frame_scores.min()),
                "score_mean": float(frame_scores.mean()),
                "score_max": float(frame_scores.max()),
            }
        )
        print(f"video {video_id}: frame_auc={per_video[-1]['frame_roc_auc']} pixel_auc_sampled={per_video[-1]['pixel_roc_auc_sampled']}")

    frame_labels_all = np.concatenate(all_frame_labels)
    frame_scores_all = np.concatenate(all_frame_scores)
    pixel_labels_all = np.concatenate(all_pixel_labels)
    pixel_scores_all = np.concatenate(all_pixel_scores)

    summary = {
        "videos": 21,
        "frames": int(len(frame_labels_all)),
        "anomaly_frames": int(frame_labels_all.sum()),
        "frame_roc_auc_eval_style": safe_auc(frame_labels_all, frame_scores_all),
        "frame_average_precision": safe_ap(frame_labels_all, frame_scores_all),
        "sampled_pixels": int(len(pixel_labels_all)),
        "sampled_positive_pixels": int(pixel_labels_all.sum()),
        "pixel_roc_auc_sampled": safe_auc(pixel_labels_all, pixel_scores_all),
        "pixel_average_precision_sampled": safe_ap(pixel_labels_all, pixel_scores_all),
        "note": "Pixel metrics use sampled pixels from resized compact spatial grids. This is not literal EVAL RBDC/TBDC.",
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with args.out_per_video.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        writer.writerows(per_video)

    print(f"Wrote summary -> {args.out_summary}")
    print(f"Wrote per-video -> {args.out_per_video}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
