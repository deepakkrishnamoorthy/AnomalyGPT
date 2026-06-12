"""Render a local inference video with anomaly heatmap and GT mask overlay.

This is an offline command-line inference visualizer. It consumes the already
computed model outputs:

- per-volume exemplar scores
- compact per-frame spatial score maps
- extracted test frames
- optional Avenue spatial GT masks

It writes a playable overlay video plus a text/JSON explanation summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio


def video_id_text(value: str | int) -> str:
    return f"{int(value):02d}"


def load_gt_masks(mask_dir: Path, video_id: str) -> list[np.ndarray] | None:
    mask_path = mask_dir / f"{int(video_id)}_label.mat"
    if not mask_path.exists():
        return None
    mat = sio.loadmat(mask_path, squeeze_me=False)
    if "volLabel" not in mat:
        return None
    return [(np.asarray(cell) > 0).astype(np.uint8) for cell in mat["volLabel"].reshape(-1)]


def normalize_heatmap(score_grid: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(score_grid, dtype=np.float32)
    return np.clip((score_grid - low) / (high - low), 0.0, 1.0).astype(np.float32)


def overlay_heatmap(frame: np.ndarray, score_grid: np.ndarray, low: float, high: float, alpha: float) -> np.ndarray:
    norm = normalize_heatmap(score_grid, low, high)
    heat_small = (norm * 255).astype(np.uint8)
    heat = cv2.resize(heat_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    alpha_map = (heat.astype(np.float32) / 255.0) * alpha
    alpha_map = alpha_map[:, :, None]
    blended = (frame.astype(np.float32) * (1.0 - alpha_map) + color.astype(np.float32) * alpha_map).astype(np.uint8)
    return blended


def overlay_gt_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None or not mask.any():
        return frame
    out = frame.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    mask3 = np.repeat(mask[:, :, None].astype(bool), 3, axis=2)
    out[mask3] = cv2.addWeighted(out, 0.72, green, 0.28, 0)[mask3]
    return out


def draw_top_region(frame: np.ndarray, row: pd.Series | None, *, box_threshold: float) -> np.ndarray:
    if row is None:
        return frame
    if float(row["anomaly_score"]) < box_threshold:
        return frame
    out = frame.copy()
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(
        out,
        f"top region {int(row['region_id'])} score {float(row['anomaly_score']):.2f}",
        (x + 4, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_hud(
    frame: np.ndarray,
    *,
    video_id: str,
    frame_idx: int,
    frame_score: float,
    gt_on: bool,
    top_reason: str,
    top_row: pd.Series | None,
    heat_floor: float,
) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 72), (0, 0, 0), -1)
    gt_text = "GT ON" if gt_on else "GT OFF"
    gt_color = (70, 255, 70) if gt_on else (210, 210, 210)
    line1 = f"Test {video_id} | F{frame_idx:04d} | frame score {frame_score:.3f} | {gt_text}"
    if top_row is not None:
        line2 = f"top region {int(top_row['region_id'])} score {float(top_row['anomaly_score']):.3f} | reason {top_reason} | heat shown above {heat_floor:.2f}"
    else:
        line2 = f"no active region | heat shown above {heat_floor:.2f}"
    cv2.putText(out, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, gt_text, (out.shape[1] - 92, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, gt_color, 2, cv2.LINE_AA)
    cv2.putText(out, line2[:88], (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def reason_from_row(row: pd.Series) -> str:
    parts = {
        "appearance": float(row["distance_app"]),
        "direction": float(row["distance_ang"]),
        "speed": float(row["distance_speed"]),
        "background": float(row["distance_bkg"]),
    }
    return max(parts, key=parts.get)


def explanation_text(video_id: str, top_row: pd.Series, gt_available: bool, gt_frames: int) -> str:
    reason = reason_from_row(top_row)
    frame_range = f"{int(top_row['start_frame'])}-{int(top_row['end_frame'])}"
    region = int(top_row["region_id"])
    score = float(top_row["anomaly_score"])
    base = (
        f"For Avenue test video {video_id}, the strongest detected anomaly is volume "
        f"{top_row['volume_id']} covering frames {frame_range} in region {region}. "
        f"The anomaly score is {score:.3f}."
    )
    if reason == "appearance":
        why = "The main reason is appearance: the visual content differs from the nearest normal exemplar in the same region."
    elif reason == "direction":
        why = "The main reason is motion direction: the movement pattern differs from the nearest normal exemplar in the same region."
    elif reason == "speed":
        why = "The main reason is speed: the motion magnitude differs from the nearest normal exemplar in the same region."
    else:
        why = "The main reason is background/stationary behavior: the motion-vs-background pattern differs from normal."
    contrib = (
        f"Distance breakdown: appearance={float(top_row['distance_app']):.3f}, "
        f"direction={float(top_row['distance_ang']):.3f}, speed={float(top_row['distance_speed']):.3f}, "
        f"background={float(top_row['distance_bkg']):.3f}. "
        f"The nearest normal exemplar is {top_row['nearest_exemplar_volume_id']}."
    )
    gt = (
        f"Spatial GT masks were available and mark {gt_frames} frames as anomalous in this video."
        if gt_available
        else "Spatial GT masks were not available for this video, so only model heatmaps are shown."
    )
    return f"{base} {why} {contrib} {gt}"


def top_row_for_frame(group: pd.DataFrame, frame_idx: int) -> pd.Series | None:
    active = group[(group["start_frame"] <= frame_idx) & (group["end_frame"] >= frame_idx)]
    if active.empty:
        return None
    return active.sort_values("anomaly_score", ascending=False).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="01")
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames/Test"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument("--spatial-map-dir", type=Path, default=Path("outputs/avenue_eval10_spatial_score_maps"))
    parser.add_argument("--mask-dir", type=Path, default=Path("Avenue Dataset/avenue-spatial-GT/ground_truth_demo/testing_label_mask"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/local_inference"))
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--heat-alpha", type=float, default=0.42)
    parser.add_argument("--heat-floor", type=float, default=0.85, help="Scores below this are transparent in the heatmap.")
    parser.add_argument("--heat-ceil-percentile", type=float, default=99.0)
    parser.add_argument("--box-threshold", type=float, default=0.90, help="Only draw top-region boxes above this score.")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional quick preview limit.")
    args = parser.parse_args()

    vid = video_id_text(args.video_id)
    frame_dir = args.frames_root / vid
    if not frame_dir.exists():
        raise RuntimeError(f"Missing extracted frames: {frame_dir}")

    scores = pd.read_csv(args.scores)
    video_scores = scores[scores["video_id"].astype(int) == int(vid)].copy()
    if video_scores.empty:
        raise RuntimeError(f"No score rows found for video {vid}")

    spatial_path = args.spatial_map_dir / f"{vid}.npy"
    if not spatial_path.exists():
        raise RuntimeError(f"Missing spatial score map {spatial_path}. Run scripts/13_build_frame_scores.py first.")
    spatial = np.load(spatial_path)

    masks = load_gt_masks(args.mask_dir, vid)
    gt_available = masks is not None
    gt_frames = int(sum(1 for mask in masks if mask.any())) if masks is not None else 0

    if args.start_frame is not None or args.end_frame is not None:
        focus_start = args.start_frame or 1
        focus_end = args.end_frame or int(video_scores["end_frame"].max())
        candidate_scores = video_scores[
            (video_scores["start_frame"] <= focus_end)
            & (video_scores["end_frame"] >= focus_start)
        ]
        if candidate_scores.empty:
            candidate_scores = video_scores
    else:
        candidate_scores = video_scores
    top_row = candidate_scores.sort_values("anomaly_score", ascending=False).iloc[0]
    reason = reason_from_row(top_row)
    explanation = explanation_text(vid, top_row, gt_available, gt_frames)

    low = float(args.heat_floor)
    high = float(np.percentile(spatial, args.heat_ceil_percentile))
    if high <= low:
        high = float(spatial.max())
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame or 1
        end_frame = args.end_frame or len(frame_paths)
        frame_paths = frame_paths[start_frame - 1 : end_frame]
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first frame {frame_paths[0]}")
    h, w = first.shape[:2]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.start_frame is not None or args.end_frame is not None:
        suffix = f"_f{args.start_frame or 1:04d}_{args.end_frame or len(frame_paths):04d}"
    out_video = args.out_dir / f"avenue_test_{vid}{suffix}_inference_overlay.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    first_frame_idx = args.start_frame or 1
    for offset, path in enumerate(frame_paths):
        idx = first_frame_idx + offset
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if idx <= spatial.shape[0]:
            frame_score = float(spatial[idx - 1].max())
            rendered = overlay_heatmap(frame, spatial[idx - 1], low, high, args.heat_alpha)
        else:
            frame_score = 0.0
            rendered = frame
        mask = masks[idx - 1] if masks is not None and idx <= len(masks) else None
        rendered = overlay_gt_mask(rendered, mask)
        active_top = top_row_for_frame(video_scores, idx)
        active_reason = reason_from_row(active_top) if active_top is not None else "none"
        rendered = draw_top_region(rendered, active_top, box_threshold=args.box_threshold)
        rendered = draw_hud(
            rendered,
            video_id=vid,
            frame_idx=idx,
            frame_score=frame_score,
            gt_on=mask is not None and mask.any(),
            top_reason=active_reason,
            top_row=active_top,
            heat_floor=args.heat_floor,
        )
        writer.write(rendered)
        if offset and offset % 250 == 0:
            print(f"Rendered {offset} frames...")
    writer.release()

    summary = {
        "video_id": vid,
        "frames_rendered": len(frame_paths),
        "render_start_frame": first_frame_idx,
        "render_end_frame": first_frame_idx + len(frame_paths) - 1,
        "heat_floor": args.heat_floor,
        "box_threshold": args.box_threshold,
        "output_video": str(out_video),
        "top_volume_id": str(top_row["volume_id"]),
        "top_score": float(top_row["anomaly_score"]),
        "top_region_id": int(top_row["region_id"]),
        "top_frame_range": [int(top_row["start_frame"]), int(top_row["end_frame"])],
        "nearest_exemplar_volume_id": str(top_row["nearest_exemplar_volume_id"]),
        "main_reason": reason,
        "gt_masks_available": gt_available,
        "gt_anomaly_frames": gt_frames,
        "explanation": explanation,
    }
    summary_path = args.out_dir / f"avenue_test_{vid}_inference_summary.json"
    text_path = args.out_dir / f"avenue_test_{vid}_inference_explanation.txt"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(explanation + "\n", encoding="utf-8")

    print(f"Wrote inference video -> {out_video}")
    print(f"Wrote summary -> {summary_path}")
    print(f"Wrote explanation -> {text_path}")
    print(explanation)


if __name__ == "__main__":
    main()
