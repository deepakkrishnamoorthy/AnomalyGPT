"""FastAPI dashboard for AnomalyGPT offline explanation cases."""

from __future__ import annotations

import csv
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
EXPLANATION_JSON = PROJECT_ROOT / "outputs" / "explanations" / "top_anomaly_cases.json"
FRAME_AUC_SUMMARY = PROJECT_ROOT / "outputs" / "avenue_eval10_frame_auc_summary.json"
SPATIAL_AUC_SUMMARY = PROJECT_ROOT / "outputs" / "avenue_eval10_spatial_mask_auc_summary.json"
FRAME_SCORES_CSV = PROJECT_ROOT / "outputs" / "avenue_eval10_frame_scores.csv"
SPATIAL_MAP_DIR = PROJECT_ROOT / "outputs" / "avenue_eval10_spatial_score_maps"
FEATURES_CSV = PROJECT_ROOT / "features" / "avenue_eval10_model_features.csv"
GT_INTERVALS_MAT = PROJECT_ROOT / "Avenue Dataset" / "avenue.mat"
GT_MASK_DIR = PROJECT_ROOT / "Avenue Dataset" / "avenue-spatial-GT" / "ground_truth_demo" / "testing_label_mask"
FRAMES_ROOT = PROJECT_ROOT / "data" / "avenue_frames" / "Test"


app = FastAPI(title="AnomalyGPT Avenue Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_artifact_path(value: str | None) -> str | None:
    if not value:
        return None
    rel = Path(value.replace("\\", "/"))
    return f"/artifact/{rel.as_posix()}"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fallback_explanation(case: dict[str, Any]) -> str:
    region = case.get("region_id", "unknown")
    reason = case.get("main_reason", "score")
    if reason == "appearance":
        return (
            f"In region {region}, the test volume is anomalous mainly because its visual appearance "
            "does not match the closest normal exemplar for that same region."
        )
    if reason == "direction":
        return (
            f"In region {region}, the test volume is anomalous mainly because its motion direction "
            "differs from the closest normal exemplar for that same region."
        )
    if reason == "speed":
        return (
            f"In region {region}, the test volume is anomalous mainly because its motion speed "
            "differs from the closest normal exemplar for that same region."
        )
    return (
        f"In region {region}, the test volume is anomalous because its feature pattern differs from "
        "the closest normal exemplar for that same region."
    )


def load_cases() -> list[dict[str, Any]]:
    cases = read_json(EXPLANATION_JSON, fallback=[])
    feature_rows = load_case_feature_rows(tuple(case.get("volume_id", "") for case in cases))
    hydrated = []
    for case in cases:
        item = dict(case)
        item["plain_english_explanation"] = item.get("plain_english_explanation") or fallback_explanation(item)
        item["frame_focus"] = int(round((as_int(item.get("start_frame")) + as_int(item.get("end_frame"))) / 2))
        item["frame_range"] = f"{item.get('start_frame')}-{item.get('end_frame')}"
        item["region_box"] = {
            "x": as_int(item.get("x")),
            "y": as_int(item.get("y")),
            "w": as_int(item.get("w")),
            "h": as_int(item.get("h")),
        }
        item["component_distances"] = component_distances(item)
        item["component_badges"] = component_badges(item["component_distances"])
        item["motion_attributes"] = feature_rows.get(str(item.get("volume_id")), {})
        item["raw_frame_url"] = f"/api/cases/{item['case_id']}/frame?mode=raw"
        item["raw_overlay_url"] = item["raw_frame_url"]
        item["heatmap_url"] = f"/api/cases/{item['case_id']}/frame?mode=heatmap"
        item["heatmap_overlay_url"] = item["heatmap_url"]
        item["gt_overlay_url"] = f"/api/cases/{item['case_id']}/frame?mode=gt"
        item["combined_overlay_url"] = f"/api/cases/{item['case_id']}/frame?mode=combined"
        item["report_url"] = f"/api/cases/{item['case_id']}/report"
        for key in [
            "instrument_panel",
            "test_sheet",
            "nearest_exemplar_sheet",
            "test_clip",
            "nearest_exemplar_clip",
            "test_clip_gif",
            "nearest_exemplar_clip_gif",
        ]:
            item[f"{key}_url"] = normalize_artifact_path(item.get(key))
        hydrated.append(item)
    return hydrated


def component_distances(case: dict[str, Any]) -> dict[str, float]:
    return {
        "appearance": as_float(case.get("distance_app")),
        "direction": as_float(case.get("distance_ang")),
        "speed": as_float(case.get("distance_speed")),
        "background": as_float(case.get("distance_bkg")),
    }


def component_badges(distances: dict[str, float]) -> list[dict[str, Any]]:
    total = sum(max(0.0, value) for value in distances.values()) or 1.0
    return [
        {
            "name": name,
            "distance": value,
            "share": value / total,
            "level": "high" if value / total >= 0.4 else "medium" if value / total >= 0.2 else "low",
        }
        for name, value in distances.items()
    ]


@lru_cache(maxsize=8)
def load_case_feature_rows(volume_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    wanted = {volume_id for volume_id in volume_ids if volume_id}
    if not wanted or not FEATURES_CSV.exists():
        return {}

    keep = {
        "volume_id",
        "motion_background_cls",
        "motion_stationary_fraction",
        "motion_moving_fraction",
        "motion_mean_magnitude",
        "motion_max_magnitude",
    }
    keep.update({f"motion_angle_hist_{idx:02d}" for idx in range(12)})
    keep.update({f"motion_speed_{idx:02d}" for idx in range(12)})

    rows: dict[str, dict[str, Any]] = {}
    with FEATURES_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            volume_id = row.get("volume_id", "")
            if volume_id not in wanted:
                continue
            out: dict[str, Any] = {}
            for key in keep:
                if key == "volume_id":
                    continue
                out[key] = as_float(row.get(key))
            rows[volume_id] = out
            if len(rows) == len(wanted):
                break
    return rows


@lru_cache(maxsize=1)
def load_frame_scores() -> dict[str, list[dict[str, Any]]]:
    if not FRAME_SCORES_CSV.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    with FRAME_SCORES_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = str(row.get("video_id", "")).zfill(2)
            grouped.setdefault(video_id, []).append(
                {
                    "frame": as_int(row.get("frame")),
                    "score": as_float(row.get("frame_score")),
                }
            )
    return grouped


@lru_cache(maxsize=1)
def load_gt_intervals() -> dict[str, list[tuple[int, int]]]:
    if not GT_INTERVALS_MAT.exists():
        return {}
    mat = sio.loadmat(GT_INTERVALS_MAT, squeeze_me=False)
    gt = mat.get("gt")
    if gt is None:
        return {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    for idx in range(gt.shape[1]):
        cell = gt[0, idx]
        video_id = f"{idx + 1:02d}"
        video_intervals = []
        if getattr(cell, "size", 0):
            arr = np.asarray(cell)
            for start, end in arr.reshape(-1, 2):
                video_intervals.append((int(start), int(end)))
        intervals[video_id] = video_intervals
    return intervals


@lru_cache(maxsize=32)
def load_gt_masks(video_id: str) -> list[np.ndarray]:
    mask_path = GT_MASK_DIR / f"{int(video_id)}_label.mat"
    if not mask_path.exists():
        return []
    mat = sio.loadmat(mask_path, squeeze_me=False)
    cells = mat.get("volLabel")
    if cells is None:
        return []
    return [np.asarray(cell, dtype=np.uint8) for cell in cells.reshape(-1)]


def safe_project_path(path_text: str) -> Path:
    rel = Path(path_text.replace("\\", "/"))
    target = (PROJECT_ROOT / rel).resolve()
    root = PROJECT_ROOT.resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=403, detail="Artifact path escapes project root.")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path_text}")
    return target


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "cases_available": EXPLANATION_JSON.exists(),
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return {
        "frame_auc": read_json(FRAME_AUC_SUMMARY, fallback={}),
        "spatial_mask_auc": read_json(SPATIAL_AUC_SUMMARY, fallback={}),
    }


@app.get("/api/cases")
def cases() -> list[dict[str, Any]]:
    return load_cases()


@app.get("/api/videos")
def videos() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    frame_scores = load_frame_scores()
    for video_id, frames in frame_scores.items():
        scores = [as_float(frame.get("score")) for frame in frames]
        grouped[video_id] = {
            "video_id": video_id,
            "case_count": 0,
            "max_score": max(scores) if scores else 0.0,
            "frame_count": len(frames),
        }
    for case in load_cases():
        video_id = str(case.get("video_id", "")).zfill(2)
        entry = grouped.setdefault(
            video_id,
            {
                "video_id": video_id,
                "case_count": 0,
                "max_score": 0.0,
                "frame_count": len(frame_scores.get(video_id, [])),
            },
        )
        entry["case_count"] += 1
        entry["max_score"] = max(entry["max_score"], float(case.get("anomaly_score", 0.0)))
    return sorted(grouped.values(), key=lambda x: x["video_id"])


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str) -> dict[str, Any]:
    for case in load_cases():
        if case.get("case_id") == case_id:
            return case
    raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")


@app.get("/api/videos/{video_id}/timeline")
def video_timeline(video_id: str) -> dict[str, Any]:
    video_id = video_id.zfill(2)
    frames = load_frame_scores().get(video_id, [])
    if not frames:
        raise HTTPException(status_code=404, detail=f"No frame scores for video {video_id}")
    scores = [frame["score"] for frame in frames]
    max_score = max(scores) if scores else 1.0
    cases_for_video = [
        {
            "case_id": case["case_id"],
            "start_frame": case["start_frame"],
            "end_frame": case["end_frame"],
            "frame_focus": case["frame_focus"],
            "score": case["anomaly_score"],
            "reason": case["main_reason"],
            "region_id": case["region_id"],
        }
        for case in load_cases()
        if str(case.get("video_id", "")).zfill(2) == video_id
    ]
    return {
        "video_id": video_id,
        "frames": frames,
        "score_min": min(scores),
        "score_max": max_score,
        "score_mean": sum(scores) / len(scores),
        "gt_intervals": [{"start": start, "end": end} for start, end in load_gt_intervals().get(video_id, [])],
        "cases": cases_for_video,
    }


@app.get("/api/cases/{case_id}/report")
def case_report(case_id: str) -> dict[str, Any]:
    case = case_detail(case_id)
    return {
        "case_id": case["case_id"],
        "video_id": case["video_id"],
        "volume_id": case["volume_id"],
        "frame_range": case["frame_range"],
        "region_id": case["region_id"],
        "anomaly_score": case["anomaly_score"],
        "main_reason": case["main_reason"],
        "component_distances": case["component_distances"],
        "explanation": case["plain_english_explanation"],
    }


@app.get("/api/cases/{case_id}/frame")
def case_frame(
    case_id: str,
    mode: str = Query("combined", pattern="^(raw|heatmap|gt|combined)$"),
) -> Response:
    case = case_detail(case_id)
    frame = render_case_frame(case, mode)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


def render_case_frame(case: dict[str, Any], mode: str) -> np.ndarray:
    video_id = str(case.get("video_id", "")).zfill(2)
    frame_idx = as_int(case.get("frame_focus")) or as_int(case.get("start_frame"))
    frame_path = FRAMES_ROOT / video_id / f"{frame_idx:05d}.jpg"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Missing frame: {frame_path}")

    out = frame.copy()
    if mode in {"heatmap", "combined"}:
        out = overlay_heatmap(out, video_id, frame_idx)
    if mode in {"gt", "combined"}:
        out = overlay_gt_mask(out, video_id, frame_idx)

    box = case.get("region_box", {})
    x, y, w, h = as_int(box.get("x")), as_int(box.get("y")), as_int(box.get("w")), as_int(box.get("h"))
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(
        out,
        f"region {case.get('region_id')} score {as_float(case.get('anomaly_score')):.2f}",
        (max(8, x), max(24, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def overlay_heatmap(frame: np.ndarray, video_id: str, frame_idx: int) -> np.ndarray:
    map_path = SPATIAL_MAP_DIR / f"{video_id}.npy"
    if not map_path.exists():
        return frame
    maps = np.load(map_path)
    if frame_idx < 1 or frame_idx > maps.shape[0]:
        return frame
    grid = maps[frame_idx - 1].astype(np.float32)
    if float(grid.max()) <= 0.0:
        return frame
    norm = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.resize(norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    mask = heat > 8
    blended = cv2.addWeighted(frame, 0.62, heat_color, 0.38, 0)
    frame[mask] = blended[mask]
    return frame


def overlay_gt_mask(frame: np.ndarray, video_id: str, frame_idx: int) -> np.ndarray:
    masks = load_gt_masks(video_id)
    if frame_idx < 1 or frame_idx > len(masks):
        return frame
    mask = masks[frame_idx - 1]
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    active = mask.astype(bool)
    if not np.any(active):
        return frame
    overlay = frame.copy()
    overlay[active] = (40, 40, 230)
    frame[active] = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)[active]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)
    return frame


@app.get("/artifact/{artifact_path:path}")
def artifact(artifact_path: str) -> FileResponse:
    target = safe_project_path(artifact_path)
    return FileResponse(target)
