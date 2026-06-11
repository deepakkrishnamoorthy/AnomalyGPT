# AnomalyGPT Deployment App

FastAPI dashboard for offline Avenue anomaly explanation results.

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
outputs\avenue_eval10_frame_auc_summary.json
outputs\avenue_eval10_spatial_mask_auc_summary.json
```

Regenerate explanations whenever you want the dashboard to show a different subset of cases.
