"""Compute lightweight motion pseudo-tracks and physics features for Avenue clips."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import cv2


@dataclass
class Detection:
    frame: int
    bbox: tuple[int, int, int, int]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class MotionTrack:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]


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


def center_distance(left: Detection, right: Detection) -> float:
    lx, ly = left.center
    rx, ry = right.center
    return math.hypot(lx - rx, ly - ry)


def detect_motion(frame, subtractor, *, min_area: int, max_area: int, frame_id: int) -> list[Detection]:
    fg = subtractor.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.dilate(fg, kernel, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 8:
            continue
        detections.append(
            Detection(
                frame=frame_id,
                bbox=(int(x), int(y), int(x + w), int(y + h)),
                score=round(min(area / frame_area, 1.0), 6),
            )
        )
    return sorted(detections, key=lambda item: item.score, reverse=True)


def link_tracks(
    frames_detections: list[list[Detection]],
    *,
    max_distance: float,
    max_missed: int,
    min_track_len: int,
) -> list[MotionTrack]:
    next_track_id = 1
    active: list[MotionTrack] = []
    finished: list[MotionTrack] = []

    for detections in frames_detections:
        unmatched = detections[:]

        for track in active:
            if not unmatched:
                track.missed += 1
                continue
            distances = [(center_distance(track.last_detection, det), idx, det) for idx, det in enumerate(unmatched)]
            distance, idx, detection = min(distances, key=lambda item: item[0])
            if distance <= max_distance:
                track.detections.append(detection)
                track.missed = 0
                unmatched.pop(idx)
            else:
                track.missed += 1

        still_active: list[MotionTrack] = []
        for track in active:
            if track.missed > max_missed:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        for detection in unmatched:
            active.append(MotionTrack(track_id=next_track_id, detections=[detection]))
            next_track_id += 1

    finished.extend(active)
    return [track for track in finished if len(track.detections) >= min_track_len]


def angle_deviation(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def compute_track_features(track: MotionTrack, fps: float) -> dict:
    centers = [det.center for det in track.detections]
    frame_ids = [det.frame for det in track.detections]
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
    direction = mean(directions) if directions else 0.0
    mean_velocity = mean(velocities) if velocities else 0.0
    max_velocity = max(velocities) if velocities else 0.0
    acceleration = mean(accelerations) if accelerations else 0.0
    stationary_fraction = (
        sum(1 for velocity in velocities if velocity < 5.0) / len(velocities)
        if velocities
        else 1.0
    )

    turn_angles = [
        angle_deviation(curr, prev)
        for prev, curr in zip(directions, directions[1:])
    ]
    curvature = mean(turn_angles) if turn_angles else 0.0

    return {
        "track_id": track.track_id,
        "frame_start": frame_ids[0],
        "frame_end": frame_ids[-1],
        "track_length": len(track.detections),
        "mean_velocity_px_s": round(mean_velocity, 6),
        "max_velocity_px_s": round(max_velocity, 6),
        "mean_acceleration_px_s2": round(acceleration, 6),
        "direction_deg": round(direction, 6),
        "curvature_deg": round(curvature, 6),
        "stationary_fraction": round(stationary_fraction, 6),
    }


def compute_clip_physics(tracks: list[MotionTrack], fps: float, frame_labels: list[int]) -> dict:
    features = [compute_track_features(track, fps) for track in tracks]
    moving = [item for item in features if item["track_length"] >= 2]
    mean_speed = mean([item["mean_velocity_px_s"] for item in moving]) if moving else 0.0
    mean_direction = mean([item["direction_deg"] for item in moving]) if moving else 0.0

    for item in features:
        item["velocity_ratio"] = round(item["mean_velocity_px_s"] / (mean_speed + 1e-6), 6) if mean_speed else 0.0
        item["flow_deviation_deg"] = round(angle_deviation(item["direction_deg"], mean_direction), 6)
        item["overlaps_anomaly_frames"] = any(
            frame_labels[frame_id - 1] == 1
            for frame_id in range(item["frame_start"], item["frame_end"] + 1)
            if 0 <= frame_id - 1 < len(frame_labels)
        )
        item["is_candidate_anomaly"] = bool(
            item["overlaps_anomaly_frames"]
            and (
                item["velocity_ratio"] >= 1.5
                or item["flow_deviation_deg"] >= 75.0
                or item["curvature_deg"] >= 60.0
            )
        )

    return {
        "features": features,
        "crowd_flow": {
            "mean_speed_px_s": round(mean_speed, 6),
            "mean_direction_deg": round(mean_direction, 6),
            "track_count": len(features),
        },
    }


def track_clip(clip: dict, *, min_area: int, max_area: int, max_distance: float, max_missed: int, min_track_len: int) -> tuple[dict, dict]:
    subtractor = cv2.createBackgroundSubtractorMOG2(history=32, varThreshold=24, detectShadows=False)
    frames_detections: list[list[Detection]] = []
    for frame_id, frame_path in enumerate(clip["frame_paths"], start=1):
        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(frame_path)
        detections = detect_motion(
            frame,
            subtractor,
            min_area=min_area,
            max_area=max_area,
            frame_id=frame_id,
        )
        frames_detections.append(detections)

    tracks = link_tracks(
        frames_detections,
        max_distance=max_distance,
        max_missed=max_missed,
        min_track_len=min_track_len,
    )
    track_payload = {
        "clip_id": clip["clip_id"],
        "dataset": "avenue",
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "computed_motion_pseudo_tracks",
        "method": "opencv_mog2_contour_centroid_linking",
        "tracks": [
            {
                "track_id": track.track_id,
                "category": "motion_blob",
                "boxes": [
                    {
                        "frame": det.frame,
                        "bbox": list(det.bbox),
                        "score": det.score,
                    }
                    for det in track.detections
                ],
            }
            for track in tracks
        ],
    }

    physics = compute_clip_physics(tracks, clip["fps"], clip["frame_labels"])
    physics_payload = {
        "clip_id": clip["clip_id"],
        "dataset": "avenue",
        "video_id": clip["video_id"],
        "split": clip["split"],
        "fps": clip["fps"],
        "status": "computed_motion_pseudo_physics",
        "method": "opencv_mog2_contour_centroid_linking",
        "features": physics["features"],
        "crowd_flow": physics["crowd_flow"],
        "ground_truth": {
            "clip_label": clip["clip_label"],
            "anomaly_frame_count": clip["anomaly_frame_count"],
            "anomaly_overlap": clip["anomaly_overlap"],
        },
    }
    return track_payload, physics_payload


def summarize(processed: list[dict]) -> dict:
    total_tracks = sum(item["track_count"] for item in processed)
    candidate_tracks = sum(item["candidate_track_count"] for item in processed)
    clips_with_tracks = sum(1 for item in processed if item["track_count"] > 0)
    return {
        "clips_in_summary": len(processed),
        "clips_with_tracks": clips_with_tracks,
        "total_tracks": total_tracks,
        "candidate_anomaly_tracks": candidate_tracks,
        "mean_tracks_per_clip": round(total_tracks / len(processed), 6) if processed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-manifest", type=Path, default=Path("manifests/anomalygpt_avenue_clips.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/avenue_motion_physics_summary.json"))
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=60000)
    parser.add_argument("--max-distance", type=float, default=55.0)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute clips even if physics is already present.")
    args = parser.parse_args()

    clips = load_jsonl(args.clip_manifest)
    if args.split != "all":
        clips = [clip for clip in clips if clip["split"] == args.split]
    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    processed: list[dict] = []
    skipped = 0
    for index, clip in enumerate(clips, start=1):
        physics_path = Path(clip["physics_path"])
        tracks_path = Path(clip["tracks_path"])
        physics = None
        tracks = None
        if not args.force and physics_path.exists() and tracks_path.exists():
            try:
                existing_physics = json.loads(physics_path.read_text(encoding="utf-8"))
                existing_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
                if existing_physics.get("status") == "computed_motion_pseudo_physics":
                    physics = existing_physics
                    tracks = existing_tracks
                    skipped += 1
            except json.JSONDecodeError:
                physics = None
                tracks = None

        if physics is None or tracks is None:
            tracks, physics = track_clip(
                clip,
                min_area=args.min_area,
                max_area=args.max_area,
                max_distance=args.max_distance,
                max_missed=args.max_missed,
                min_track_len=args.min_track_len,
            )
            write_json(tracks_path, tracks)
            write_json(physics_path, physics)

        processed.append(
            {
                "clip_id": clip["clip_id"],
                "split": clip["split"],
                "clip_label": clip["clip_label"],
                "track_count": len(tracks["tracks"]),
                "candidate_track_count": sum(
                    1 for item in physics["features"] if item["is_candidate_anomaly"]
                ),
            }
        )
        if index % 100 == 0:
            print(f"Scanned {index}/{len(clips)} clips; resumed/skipped {skipped}")

    payload = {
        "method": "opencv_mog2_contour_centroid_linking",
        "parameters": {
            "min_area": args.min_area,
            "max_area": args.max_area,
            "max_distance": args.max_distance,
            "max_missed": args.max_missed,
            "min_track_len": args.min_track_len,
        },
        "summary": summarize(processed),
        "skipped_existing_computed": skipped,
        "clips": processed,
    }
    write_json(args.summary_out, payload)
    print(f"Processed clips: {len(processed)}")
    print(f"Wrote: {args.summary_out}")


if __name__ == "__main__":
    main()
