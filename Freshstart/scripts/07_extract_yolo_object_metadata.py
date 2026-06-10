"""Extract YOLO/ByteTrack object metadata for EVAL-style volumes.

This stage fills the object columns that the lightweight instrument feature
script currently leaves as placeholders. It detects objects on full frames once,
then aggregates detections into each manifest region and 10-frame time window.

COCO-to-EVAL mapping:
- person -> object_person
- car, bus, truck -> object_car
- bicycle, motorcycle -> object_cyclist
- dog -> object_dog

Tree/house/skyscraper/bridge are not COCO classes in standard YOLO weights, so
they remain zero unless a custom detector is used later.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}

OBJECT_COLUMNS = [
    "object_person",
    "object_car",
    "object_cyclist",
    "object_dog",
    "object_tree",
    "object_house",
    "object_skyscraper",
    "object_bridge",
    "object_unknown_motion",
]

COCO_TO_OBJECT = {
    "person": "object_person",
    "car": "object_car",
    "bus": "object_car",
    "truck": "object_car",
    "bicycle": "object_cyclist",
    "motorcycle": "object_cyclist",
    "dog": "object_dog",
}


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics.\n"
            "Install it before running this stage, for example:\n"
            "  pip install ultralytics\n"
            "Then run this script again with a YOLO model path such as yolov8n.pt."
        ) from exc
    return YOLO


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_selected(row: dict, split: str) -> bool:
    return split == "all" or row["split"] == split


def detection_overlaps_region(det: dict, row: dict, min_overlap: float) -> bool:
    x1, y1, x2, y2 = det["xyxy"]
    rx1 = float(row["x"])
    ry1 = float(row["y"])
    rx2 = rx1 + float(row["w"])
    ry2 = ry1 + float(row["h"])

    inter_w = max(0.0, min(x2, rx2) - max(x1, rx1))
    inter_h = max(0.0, min(y2, ry2) - max(y1, ry1))
    inter = inter_w * inter_h
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    return (inter / box_area) >= min_overlap


def parse_result(result, names: dict[int, str], conf_threshold: float) -> list[dict]:
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.detach().cpu().numpy().astype(int)

    for idx, box in enumerate(xyxy):
        if float(conf[idx]) < conf_threshold:
            continue
        class_name = names.get(int(cls[idx]), str(int(cls[idx])))
        detections.append(
            {
                "xyxy": [float(v) for v in box],
                "class_id": int(cls[idx]),
                "class_name": class_name,
                "confidence": float(conf[idx]),
                "track_id": int(ids[idx]) if ids is not None else None,
            }
        )
    return detections


def detect_required_frames(
    rows: list[dict],
    *,
    frames_root: Path,
    model,
    conf_threshold: float,
    image_size: int,
    use_tracking: bool,
    tracker: str,
) -> dict[tuple[str, str, int], list[dict]]:
    required = defaultdict(set)
    for row in rows:
        key = (row["split"], row["video_id"])
        required[key].update(range(int(row["start_frame"]), int(row["end_frame"]) + 1))

    detections_by_frame: dict[tuple[str, str, int], list[dict]] = {}
    names = model.names

    for (split, video_id), frame_indices in sorted(required.items()):
        frame_dir = frames_root / SPLIT_TO_FRAME_DIR[split] / video_id
        print(f"Detecting {split}/{video_id}: {len(frame_indices)} frames")

        if use_tracking:
            # Reset tracker state between videos.
            model.predict(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False)

        for frame_idx in sorted(frame_indices):
            frame_path = frame_dir / f"{frame_idx:05d}.jpg"
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read frame {frame_path}")

            if use_tracking:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker,
                    conf=conf_threshold,
                    imgsz=image_size,
                    verbose=False,
                )
            else:
                results = model.predict(frame, conf=conf_threshold, imgsz=image_size, verbose=False)

            detections_by_frame[(split, video_id, frame_idx)] = parse_result(results[0], names, conf_threshold)

    return detections_by_frame


def aggregate_row(row: dict, detections_by_frame: dict, min_overlap: float) -> dict:
    object_hits = {column: 0.0 for column in OBJECT_COLUMNS}
    object_counts = {f"{column}_count": 0 for column in OBJECT_COLUMNS if column != "object_unknown_motion"}
    confidences = defaultdict(list)
    track_ids = defaultdict(set)

    total_detections = 0
    mapped_detections = 0

    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame_detections = detections_by_frame.get((row["split"], row["video_id"], frame_idx), [])
        for det in frame_detections:
            if not detection_overlaps_region(det, row, min_overlap=min_overlap):
                continue
            total_detections += 1
            object_column = COCO_TO_OBJECT.get(det["class_name"])
            if object_column is None:
                continue

            mapped_detections += 1
            object_hits[object_column] = 1.0
            object_counts[f"{object_column}_count"] += 1
            confidences[object_column].append(det["confidence"])
            if det["track_id"] is not None:
                track_ids[object_column].add(det["track_id"])

    # This means "YOLO did not map any known object into this volume".
    object_hits["object_unknown_motion"] = 1.0 if mapped_detections == 0 else 0.0

    out = dict(row)
    out.update(object_hits)
    out.update(object_counts)
    out["yolo_total_region_detections"] = total_detections
    out["yolo_mapped_region_detections"] = mapped_detections
    for column in ["object_person", "object_car", "object_cyclist", "object_dog"]:
        conf_values = confidences[column]
        out[f"{column}_max_conf"] = max(conf_values) if conf_values else 0.0
        out[f"{column}_mean_conf"] = float(np.mean(conf_values)) if conf_values else 0.0
        out[f"{column}_track_count"] = len(track_ids[column])
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/avenue_eval10_volume_manifest.jsonl"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-csv", type=Path, default=Path("features/avenue_eval10_yolo_object_metadata.csv"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("manifests/avenue_eval10_yolo_object_metadata.jsonl"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--split", choices=["training", "testing", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-overlap", type=float, default=0.20)
    parser.add_argument("--use-tracking", action="store_true", help="Use YOLO + ByteTrack instead of frame-only YOLO.")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    args = parser.parse_args()

    selected_rows = []
    for row in iter_manifest(args.manifest):
        if not row_selected(row, args.split):
            continue
        selected_rows.append(row)
        if args.limit is not None and len(selected_rows) >= args.limit:
            break

    if not selected_rows:
        raise SystemExit("No manifest rows selected.")

    YOLO = require_ultralytics()
    model = YOLO(args.model)

    detections_by_frame = detect_required_frames(
        selected_rows,
        frames_root=args.frames_root,
        model=model,
        conf_threshold=args.conf,
        image_size=args.imgsz,
        use_tracking=args.use_tracking,
        tracker=args.tracker,
    )

    enriched_rows = [
        aggregate_row(row, detections_by_frame, min_overlap=args.min_overlap)
        for row in selected_rows
    ]
    write_csv(args.out_csv, enriched_rows)
    write_jsonl(args.out_jsonl, enriched_rows)
    print(f"Wrote {len(enriched_rows)} YOLO object metadata rows -> {args.out_csv}")
    print(f"Wrote {len(enriched_rows)} YOLO object metadata rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
