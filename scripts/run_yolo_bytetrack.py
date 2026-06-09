"""Run YOLO + ByteTrack over AnomalyGPT Avenue clips.

This script requires the optional dependency in `requirements-yolo.txt`.
It writes `tracks.json` in the same per-clip location used by the lightweight
motion pseudo-tracker, but with detector class names and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


COCO_TO_PANEL_CLASS = {
    "person": "person",
    "car": "car",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "dog": "dog",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def bbox_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def angle_deviation(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def compute_track_features(track: dict, fps: float, frame_labels: list[int]) -> dict:
    boxes = sorted(track["boxes"], key=lambda item: item["frame"])
    centers = [bbox_center(item["bbox"]) for item in boxes]
    frame_ids = [int(item["frame"]) for item in boxes]
    velocities: list[float] = []
    directions: list[float] = []

    for prev_idx in range(len(centers) - 1):
        curr_idx = prev_idx + 1
        dt = max((frame_ids[curr_idx] - frame_ids[prev_idx]) / fps, 1e-6)
        dx = centers[curr_idx][0] - centers[prev_idx][0]
        dy = centers[curr_idx][1] - centers[prev_idx][1]
        velocities.append(math.hypot(dx, dy) / dt)
        directions.append(math.degrees(math.atan2(dy, dx)))

    accelerations = [
        abs(curr - prev) * fps
        for prev, curr in zip(velocities, velocities[1:])
    ]
    turn_angles = [
        angle_deviation(curr, prev)
        for prev, curr in zip(directions, directions[1:])
    ]
    mean_velocity = mean(velocities) if velocities else 0.0
    max_velocity = max(velocities) if velocities else 0.0
    acceleration = mean(accelerations) if accelerations else 0.0
    direction = mean(directions) if directions else 0.0
    curvature = mean(turn_angles) if turn_angles else 0.0
    stationary_fraction = (
        sum(1 for velocity in velocities if velocity < 5.0) / len(velocities)
        if velocities
        else 1.0
    )
    overlaps_anomaly = any(
        frame_labels[frame_id - 1] == 1
        for frame_id in range(frame_ids[0], frame_ids[-1] + 1)
        if 0 <= frame_id - 1 < len(frame_labels)
    )

    return {
        "track_id": track["track_id"],
        "category": track["category"],
        "panel_class": COCO_TO_PANEL_CLASS.get(track["category"], "unknown"),
        "frame_start": frame_ids[0],
        "frame_end": frame_ids[-1],
        "track_length": len(boxes),
        "mean_velocity_px_s": round(mean_velocity, 6),
        "max_velocity_px_s": round(max_velocity, 6),
        "mean_acceleration_px_s2": round(acceleration, 6),
        "direction_deg": round(direction, 6),
        "curvature_deg": round(curvature, 6),
        "stationary_fraction": round(stationary_fraction, 6),
        "overlaps_anomaly_frames": overlaps_anomaly,
    }


def compute_physics(clip: dict, track_payload: dict) -> dict:
    features = [
        compute_track_features(track, clip["fps"], clip["frame_labels"])
        for track in track_payload["tracks"]
        if len(track["boxes"]) >= 2
    ]
    moving = [item for item in features if item["track_length"] >= 2]
    mean_speed = mean([item["mean_velocity_px_s"] for item in moving]) if moving else 0.0
    mean_direction = mean([item["direction_deg"] for item in moving]) if moving else 0.0
    for item in features:
        item["velocity_ratio"] = round(item["mean_velocity_px_s"] / (mean_speed + 1e-6), 6) if mean_speed else 0.0
        item["flow_deviation_deg"] = round(angle_deviation(item["direction_deg"], mean_direction), 6)
        item["is_candidate_anomaly"] = bool(
            item["overlaps_anomaly_frames"]
            and (
                item["velocity_ratio"] >= 1.5
                or item["flow_deviation_deg"] >= 75.0
                or item["curvature_deg"] >= 60.0
            )
        )
    return {
        "clip_id": clip["clip_id"],
        "dataset": clip["dataset"],
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "computed_yolo_bytetrack_physics",
        "method": "ultralytics_yolo_bytetrack",
        "features": features,
        "crowd_flow": {
            "mean_speed_px_s": round(mean_speed, 6),
            "mean_direction_deg": round(mean_direction, 6),
            "track_count": len(features),
        },
        "ground_truth": {
            "clip_label": clip["clip_label"],
            "anomaly_frame_count": clip["anomaly_frame_count"],
            "anomaly_overlap": clip["anomaly_overlap"],
        },
    }


def run_clip(model, clip: dict, *, conf: float, iou: float, tracker: str, device: str | None) -> dict:
    names = model.names
    tracks_by_id: dict[int, dict] = {}
    for local_frame, frame_path in enumerate(clip["frame_paths"], start=1):
        results = model.track(
            source=frame_path,
            persist=True,
            conf=conf,
            iou=iou,
            tracker=tracker,
            device=device,
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        ids = result.boxes.id.cpu().int().tolist()
        classes = result.boxes.cls.cpu().int().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for box, track_id, cls_id, score in zip(boxes, ids, classes, scores):
            category = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            item = tracks_by_id.setdefault(
                int(track_id),
                {
                    "track_id": int(track_id),
                    "category": category,
                    "panel_class": COCO_TO_PANEL_CLASS.get(category, "unknown"),
                    "boxes": [],
                },
            )
            item["boxes"].append(
                {
                    "frame": local_frame,
                    "bbox": [round(float(v), 3) for v in box],
                    "score": round(float(score), 6),
                }
            )

    return {
        "clip_id": clip["clip_id"],
        "dataset": clip["dataset"],
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "computed_yolo_bytetrack_tracks",
        "method": "ultralytics_yolo_bytetrack",
        "tracks": list(tracks_by_id.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-manifest", type=Path, default=Path("manifests/anomalygpt_avenue_clips.jsonl"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'ultralytics'. Install with: "
            "pip install -r requirements-yolo.txt"
        ) from exc

    clips = load_jsonl(args.clip_manifest)
    if args.split != "all":
        clips = [clip for clip in clips if clip["split"] == args.split]
    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    model = YOLO(args.model)
    processed = 0
    skipped = 0
    for clip in clips:
        physics_path = Path(clip["physics_path"])
        tracks_path = Path(clip["tracks_path"])
        if not args.force and physics_path.exists():
            try:
                current = json.loads(physics_path.read_text(encoding="utf-8"))
                if current.get("status") == "computed_yolo_bytetrack_physics":
                    skipped += 1
                    continue
            except json.JSONDecodeError:
                pass

        tracks = run_clip(
            model,
            clip,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            device=args.device,
        )
        physics = compute_physics(clip, tracks)
        write_json(tracks_path, tracks)
        write_json(physics_path, physics)
        processed += 1
        if processed % 25 == 0:
            print(f"Processed {processed}; skipped {skipped}")

    print(f"Done. Processed={processed}; skipped={skipped}")


if __name__ == "__main__":
    main()
