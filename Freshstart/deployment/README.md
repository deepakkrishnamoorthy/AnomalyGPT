# AnomalyGPT Deployment App

FastAPI dashboard for offline Avenue anomaly predictions, heatmap overlays, timeline analysis, and explanation cases.

## Run

From `D:\Deepak\VAD-Folder\Freshstart`:

```powershell
python scripts\16_build_explanation_cases.py --top-k 12 --make-videos
python -m uvicorn deployment.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Data Source

The app reads:

```text
outputs\explanations\top_anomaly_cases.json
outputs\explanations\assets\...
outputs\avenue_eval10_frame_scores.csv
outputs\avenue_eval10_spatial_score_maps\*.npy
outputs\avenue_eval10_frame_auc_summary.json
outputs\avenue_eval10_spatial_mask_auc_summary.json
features\avenue_eval10_model_features.csv
Avenue Dataset\avenue.mat
Avenue Dataset\avenue-spatial-GT\ground_truth_demo\testing_label_mask
```

Regenerate explanations whenever you want the dashboard to show a different subset of cases. For more cases in the case explorer, increase `--top-k`.

For coverage across the whole Avenue test set, prefer per-video case generation:

```powershell
python scripts\16_build_explanation_cases.py --per-video-k 3 --dedupe-window 60 --make-videos
```

This creates up to 3 deduplicated explanation cases per test video instead of only selecting the strongest global peaks. Global `--top-k` often clusters in one video because the highest scores can come from the same scene.

## Current Views

- Filter cases by video and dominant reason.
- Sort by score, video/frame, or reason.
- Inspect the predicted anomaly clip, frame heatmap, GT overlay, and combined overlay.
- Read the generated English explanation.
- View score badges for appearance, direction, speed, and background contribution.
- Use the video timeline to compare model score peaks with GT anomaly intervals.
- Export the selected case as a JSON report from the dashboard.
