"""FastAPI dashboard for AnomalyGPT offline explanation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
EXPLANATION_JSON = PROJECT_ROOT / "outputs" / "explanations" / "top_anomaly_cases.json"
FRAME_AUC_SUMMARY = PROJECT_ROOT / "outputs" / "avenue_eval10_frame_auc_summary.json"
SPATIAL_AUC_SUMMARY = PROJECT_ROOT / "outputs" / "avenue_eval10_spatial_mask_auc_summary.json"


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
    hydrated = []
    for case in cases:
        item = dict(case)
        item["plain_english_explanation"] = item.get("plain_english_explanation") or fallback_explanation(item)
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
    for case in load_cases():
        video_id = str(case.get("video_id", "")).zfill(2)
        entry = grouped.setdefault(video_id, {"video_id": video_id, "case_count": 0, "max_score": 0.0})
        entry["case_count"] += 1
        entry["max_score"] = max(entry["max_score"], float(case.get("anomaly_score", 0.0)))
    return sorted(grouped.values(), key=lambda x: x["video_id"])


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str) -> dict[str, Any]:
    for case in load_cases():
        if case.get("case_id") == case_id:
            return case
    raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")


@app.get("/artifact/{artifact_path:path}")
def artifact(artifact_path: str) -> FileResponse:
    target = safe_project_path(artifact_path)
    return FileResponse(target)
