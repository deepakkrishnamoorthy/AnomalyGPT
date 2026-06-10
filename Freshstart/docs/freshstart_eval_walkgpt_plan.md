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
4. `scripts/05_export_eval_volumes.py`
5. `scripts/04_extract_lightweight_instrument_features.py`
6. `scripts/06_extract_appearance_features.py`
7. `scripts/07_extract_yolo_object_metadata.py`

The scripts are designed so you can run and validate each stage manually.

## Ablation Notes

- Current default temporal stride for 10-frame clip/volume manifest generation: `10`.
- Later ablation target: compare temporal stride values such as `1`, `5`, and `10` for accuracy, localization quality, and compute cost.
- Keep the ground-truth intervals from `avenue.mat` for evaluation only, not for normal-only training.

## Volume Export

- Raw exported volumes are saved as compressed grayscale `.npz` files.
- Visual validation previews are saved as `.jpg` contact sheets with the 10 frames placed left-to-right.
- Full export at temporal stride `10` can still be large because every time window is split into spatial regions.

## Appearance Features

- Start with a training-free ImageNet `resnet18` appearance embedding baseline.
- Crop RGB frames from `data/avenue_frames` using the manifest metadata, then average frame embeddings over each 10-frame volume.
- This replaces EVAL's supervised 8-class appearance network for the first baseline; YOLO/ByteTrack object semantics can be added later for explanations.

## Object Metadata

- Use YOLO as an optional enrichment stage to fill `object_person`, `object_car`, `object_cyclist`, and `object_dog`.
- Standard COCO YOLO weights do not provide tree, house, skyscraper, or bridge classes, so those remain zero until a custom detector is added.
- ByteTrack can be enabled for track IDs, but object detection metadata should stay separate from raw volume export because it is slower and model-dependent.
