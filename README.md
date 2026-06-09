# Physics-Grounded AnomalyGPT

Explainable video anomaly detection using grounded vision-language reasoning and trajectory dynamics.

This workspace is prepared for adapting the WalkGPT architecture from image-based pedestrian navigation into video anomaly detection. Dataset files are intentionally ignored by git; place downloaded UCSD Ped1, UCSD Ped2, and Avenue data under `data/raw/`.

`WalkGPT-main/` is the reference implementation this project is partially inspired by. See `architecture_report.md` for the concrete mapping from WalkGPT modules to the planned AnomalyGPT adaptation.

## Planned Dataset Layout

```text
data/raw/
  ucsd_ped1/
  ucsd_ped2/
  avenue/

datasets/anomalygpt/
  video_id/
    frames/
    masks/
    tracks.json
    qa.json
```

## Main Modules

- `model/physics_reasoner.py`: trajectory-derived behavior features.
- `model/physics_fusion.py`: visual-language-physics fusion scaffold.
- `utils/tracking.py`: tracker adapter boundary for ByteTrack or BoT-SORT.
- `utils/explanation_generator.py`: physics-to-language explanation rules.
- `anomaly_dataset_builder.py`: converts raw/video annotations into AnomalyGPT records.
- `train_anomalygpt.py`: training entry point placeholder.
- `evaluate_anomalygpt.py`: evaluation entry point placeholder.
