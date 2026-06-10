# Freshstart Plan: EVAL First, WalkGPT Minimal

This folder is the fresh working area for the anomaly-detection system.

## Direction

The core anomaly detector should follow the EVAL paper more closely than WalkGPT:

- streaming or long video input
- local spatio-temporal video volumes
- normal-only scene modeling
- anomaly score from distance/reconstruction error against normal behavior
- interpretable instrument-panel attributes

WalkGPT is still useful later, but as an admin-facing explanation/query layer:

- retrieve recorded anomaly clips
- summarize unusual motion/object attributes
- answer questions about anomaly history
- optionally point to frames/regions/masks

It should not be the first-stage anomaly detector.

## Compute-Conscious Simplification

EVAL uses separate appearance, direction, speed, and background networks. We will start lighter:

1. Use 10-frame video volumes like EVAL.
2. Use Avenue's normal training videos to build normal feature memory.
3. Start with lightweight optical-flow/frame-difference instrument features.
4. Add YOLO/ByteTrack object features as an optional second pass.
5. Train/evaluate simple models first:
   - nearest-exemplar distance
   - physics/instrument-panel autoencoder
6. Add learned 3D CNN attribute heads only after the baseline is validated.

## Fresh Dataset

Use only:

```text
Freshstart/Avenue Dataset/
  training_videos/
  testing_videos/
  training_vol/
  testing_vol/
```

Do not use the old extracted-frame Avenue folder from the parent project.

## Script Order

1. `scripts/01_audit_avenue_dataset.py`
2. `scripts/02_extract_avenue_frames.py`
3. `scripts/03_build_eval_volume_manifest.py`
4. `scripts/04_extract_lightweight_instrument_features.py`

The scripts are designed so you can run and validate each stage manually.

