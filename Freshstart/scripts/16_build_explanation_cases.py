"""Build offline qualitative explanation cases for anomaly detections.

This script turns model outputs into a compact EVAL-style qualitative report:

- top anomalous test volumes
- nearest normal exemplar for each selected test volume
- test/exemplar clip videos with the region box drawn
- contact sheets for quick inspection
- an instrument-panel image with score breakdown and motion attributes
- JSON and HTML indexes for offline review
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SPLIT_TO_FRAME_DIR = {
    "training": "Train",
    "testing": "Test",
}

MOTION_ANGLE_COLS = [f"motion_angle_hist_{idx:02d}" for idx in range(12)]
MOTION_SPEED_COLS = [f"motion_speed_{idx:02d}" for idx in range(12)]
MOTION_SCALAR_COLS = [
    "motion_background_cls",
    "motion_stationary_fraction",
    "motion_moving_fraction",
    "motion_mean_magnitude",
    "motion_max_magnitude",
]
FEATURE_META_COLS = [
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
]
FEATURE_USECOLS = FEATURE_META_COLS + MOTION_SCALAR_COLS + MOTION_ANGLE_COLS + MOTION_SPEED_COLS


def as_video_id(value) -> str:
    return f"{int(value):02d}"


def frame_path(frames_root: Path, split: str, video_id, frame_idx: int) -> Path:
    return frames_root / SPLIT_TO_FRAME_DIR[split] / as_video_id(video_id) / f"{frame_idx:05d}.jpg"


def read_frame(frames_root: Path, row: pd.Series, frame_idx: int) -> np.ndarray:
    path = frame_path(frames_root, str(row["split"]), row["video_id"], frame_idx)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame {path}")
    return frame


def draw_region(frame: np.ndarray, row: pd.Series, color=(0, 0, 255), label: str | None = None) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(out, label, (x + 4, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def crop_region(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pad_bottom = max(0, y + h - frame.shape[0])
    pad_right = max(0, x + w - frame.shape[1])
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame[y : y + h, x : x + w]


def make_contact_sheet(frames_root: Path, row: pd.Series, out_path: Path, *, annotate_full_frame: bool) -> str:
    tiles = []
    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame = read_frame(frames_root, row, frame_idx)
        tile = draw_region(frame, row, label=f"f{frame_idx}") if annotate_full_frame else crop_region(frame, row)
        tile = cv2.resize(tile, (160, 90), interpolation=cv2.INTER_AREA) if annotate_full_frame else cv2.resize(tile, (96, 96))
        cv2.putText(tile, str(frame_idx), (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(out_path)


def make_clip(frames_root: Path, row: pd.Series, out_path: Path, *, fps: float = 5.0) -> str:
    frames = []
    for frame_idx in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
        frame = read_frame(frames_root, row, frame_idx)
        frame = draw_region(frame, row, label=f"{row['volume_id']} f{frame_idx}")
        frames.append(frame)
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(out_path)


def draw_bar_panel(
    panel: np.ndarray,
    title: str,
    values: list[float],
    labels: list[str],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    color=(65, 105, 225),
) -> None:
    cv2.putText(panel, title, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    max_value = max(max(values), 1e-6)
    bar_gap = 4
    bar_w = max(3, (w - bar_gap * (len(values) - 1)) // len(values))
    for idx, value in enumerate(values):
        bx = x + idx * (bar_w + bar_gap)
        bh = int((value / max_value) * h)
        cv2.rectangle(panel, (bx, y + h - bh), (bx + bar_w, y + h), color, -1)
        cv2.putText(panel, labels[idx], (bx, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)


def make_instrument_panel(case: dict, test_feat: pd.Series, ex_feat: pd.Series, out_path: Path) -> str:
    panel = np.full((720, 1180, 3), 255, dtype=np.uint8)
    cv2.putText(panel, f"Case {case['case_id']}  Score {case['anomaly_score']:.3f}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Test {case['volume_id']}  nearest normal {case['nearest_exemplar_volume_id']}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1, cv2.LINE_AA)

    component_labels = ["app", "ang", "speed", "bkg"]
    component_values = [case["distance_app"], case["distance_ang"], case["distance_speed"], case["distance_bkg"]]
    draw_bar_panel(panel, "Distance breakdown", component_values, component_labels, x=40, y=130, w=360, h=160, color=(30, 90, 210))

    test_hist = [float(test_feat[col]) for col in MOTION_ANGLE_COLS]
    ex_hist = [float(ex_feat[col]) for col in MOTION_ANGLE_COLS]
    labels = [str(i) for i in range(12)]
    draw_bar_panel(panel, "Test direction histogram", test_hist, labels, x=450, y=130, w=300, h=150, color=(210, 80, 30))
    draw_bar_panel(panel, "Nearest exemplar direction histogram", ex_hist, labels, x=820, y=130, w=300, h=150, color=(60, 150, 80))

    test_speed = [float(test_feat[col]) for col in MOTION_SPEED_COLS]
    ex_speed = [float(ex_feat[col]) for col in MOTION_SPEED_COLS]
    draw_bar_panel(panel, "Test speed by direction", test_speed, labels, x=450, y=390, w=300, h=150, color=(210, 80, 30))
    draw_bar_panel(panel, "Nearest exemplar speed by direction", ex_speed, labels, x=820, y=390, w=300, h=150, color=(60, 150, 80))

    y0 = 360
    lines = [
        f"Likely reason: {case['main_reason']}",
        f"Test moving fraction: {float(test_feat['motion_moving_fraction']):.3f}",
        f"Exemplar moving fraction: {float(ex_feat['motion_moving_fraction']):.3f}",
        f"Test stationary fraction: {float(test_feat['motion_stationary_fraction']):.3f}",
        f"Exemplar stationary fraction: {float(ex_feat['motion_stationary_fraction']):.3f}",
        f"Test mean magnitude: {float(test_feat['motion_mean_magnitude']):.3f}",
        f"Exemplar mean magnitude: {float(ex_feat['motion_mean_magnitude']):.3f}",
        f"Region: {case['region_id']}  box=({case['x']},{case['y']},{case['w']},{case['h']})",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(panel, line, (40, y0 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path)


def select_cases(scores: pd.DataFrame, top_k: int, video_id: str | None, dedupe_window: int) -> pd.DataFrame:
    data = scores.copy()
    if video_id is not None:
        data = data[data["video_id"].astype(int) == int(video_id)]
    data = data.sort_values("anomaly_score", ascending=False)
    selected = []
    seen: list[tuple[int, int, int]] = []
    for _, row in data.iterrows():
        key = (int(row["video_id"]), int(row["region_id"]), int(row["start_frame"]))
        if any(key[0] == v and key[1] == r and abs(key[2] - s) <= dedupe_window for v, r, s in seen):
            continue
        selected.append(row)
        seen.append(key)
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("outputs/avenue_eval10_exemplar_scores.csv"))
    parser.add_argument("--features", type=Path, default=Path("features/avenue_eval10_model_features.csv"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/avenue_frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/explanations"))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--dedupe-window", type=int, default=30)
    parser.add_argument("--make-videos", action="store_true")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    cases_df = select_cases(scores, args.top_k, args.video_id, args.dedupe_window)
    needed_ids = set(cases_df["volume_id"].astype(str)) | set(cases_df["nearest_exemplar_volume_id"].astype(str))
    features = pd.read_csv(args.features, usecols=FEATURE_USECOLS)
    features["volume_id"] = features["volume_id"].astype(str)
    features = features[features["volume_id"].isin(needed_ids)].set_index("volume_id")

    cases = []
    assets_dir = args.out_dir / "assets"
    for idx, row in enumerate(cases_df.itertuples(index=False), start=1):
        score_row = row._asdict()
        volume_id = str(score_row["volume_id"])
        exemplar_id = str(score_row["nearest_exemplar_volume_id"])
        if volume_id not in features.index or exemplar_id not in features.index:
            raise RuntimeError(f"Missing feature rows for {volume_id} or {exemplar_id}")
        test_feat = features.loc[volume_id]
        ex_feat = features.loc[exemplar_id]

        components = {
            "appearance": float(score_row["distance_app"]),
            "direction": float(score_row["distance_ang"]),
            "speed": float(score_row["distance_speed"]),
            "background": float(score_row["distance_bkg"]),
        }
        main_reason = max(components, key=components.get)
        case_id = f"case_{idx:03d}"
        case = {
            "case_id": case_id,
            "volume_id": volume_id,
            "nearest_exemplar_volume_id": exemplar_id,
            "video_id": as_video_id(score_row["video_id"]),
            "start_frame": int(score_row["start_frame"]),
            "end_frame": int(score_row["end_frame"]),
            "region_id": int(score_row["region_id"]),
            "x": int(score_row["x"]),
            "y": int(score_row["y"]),
            "w": int(score_row["w"]),
            "h": int(score_row["h"]),
            "anomaly_score": float(score_row["anomaly_score"]),
            "distance_app": components["appearance"],
            "distance_ang": components["direction"],
            "distance_speed": components["speed"],
            "distance_bkg": components["background"],
            "main_reason": main_reason,
        }

        test_sheet = assets_dir / f"{case_id}_test_sheet.jpg"
        ex_sheet = assets_dir / f"{case_id}_nearest_exemplar_sheet.jpg"
        panel = assets_dir / f"{case_id}_instrument_panel.jpg"
        case["test_sheet"] = make_contact_sheet(args.frames_root, test_feat, test_sheet, annotate_full_frame=True)
        case["nearest_exemplar_sheet"] = make_contact_sheet(args.frames_root, ex_feat, ex_sheet, annotate_full_frame=True)
        case["instrument_panel"] = make_instrument_panel(case, test_feat, ex_feat, panel)
        if args.make_videos:
            case["test_clip"] = make_clip(args.frames_root, test_feat, assets_dir / f"{case_id}_test_clip.mp4")
            case["nearest_exemplar_clip"] = make_clip(args.frames_root, ex_feat, assets_dir / f"{case_id}_nearest_exemplar_clip.mp4")
        cases.append(case)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "top_anomaly_cases.json"
    json_path.write_text(json.dumps(cases, indent=2, sort_keys=True), encoding="utf-8")

    html_rows = []
    for case in cases:
        html_rows.append(
            f"""
            <section>
              <h2>{html.escape(case['case_id'])}: score {case['anomaly_score']:.3f}, reason {html.escape(case['main_reason'])}</h2>
              <p>Test {html.escape(case['volume_id'])} vs nearest normal {html.escape(case['nearest_exemplar_volume_id'])}</p>
              <img src="{Path(case['instrument_panel']).relative_to(args.out_dir).as_posix()}" />
              <h3>Test volume</h3>
              <img src="{Path(case['test_sheet']).relative_to(args.out_dir).as_posix()}" />
              <h3>Nearest normal exemplar</h3>
              <img src="{Path(case['nearest_exemplar_sheet']).relative_to(args.out_dir).as_posix()}" />
            </section>
            """
        )
    html_doc = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Avenue Explanation Cases</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f6f6f4; color: #222; }
        section { margin: 0 0 36px; padding: 18px; background: white; border: 1px solid #ddd; }
        img { max-width: 100%; display: block; margin: 8px 0 16px; border: 1px solid #ccc; }
      </style>
    </head>
    <body>
      <h1>Avenue Offline Explanation Cases</h1>
      """ + "\n".join(html_rows) + """
    </body>
    </html>
    """
    html_path = args.out_dir / "index.html"
    html_path.write_text(html_doc, encoding="utf-8")

    print(f"Wrote {len(cases)} explanation cases -> {json_path}")
    print(f"Wrote HTML report -> {html_path}")


if __name__ == "__main__":
    main()
