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
8. `scripts/08_extract_eval_motion_attributes.py`
9. `scripts/09_build_model_feature_table.py`
10. `scripts/10_train_region_exemplars.py`
11. `scripts/11_score_region_exemplars.py`

The scripts are designed so you can run and validate each stage manually.

## Ablation Notes

- Current default temporal stride for 10-frame clip/volume manifest generation: `10`.
- Later ablation target: compare temporal stride values such as `1`, `5`, and `10` for accuracy, localization quality, and compute cost.
- Current stride-10 results should be reported as an EVAL-inspired lightweight baseline, not as exact EVAL reproduction.
- Future matched-protocol experiment: rerun manifest/features/scoring with temporal stride `1` or the closest EVAL-compatible temporal sampling before comparing directly to EVAL paper numbers.
- Current appearance baseline: ImageNet `resnet18` mean-pooled over the 10-frame volume.
- Later appearance ablation target: compare `resnet18` against DINOv3-style self-supervised vision features for accuracy vs. compute cost.
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

## Motion Attributes

- Start with direct optical-flow attributes instead of training EVAL's 3D CNN motion models immediately.
- Extract `Yang` as a 12-bin direction histogram, `Yspeed` as 12 average speed values, `Ybkg.pix` as stationary pixel fraction, and `Ybkg.cls` as a background/no-motion label.
- Use these attributes directly for the first normal-only anomaly baseline, then later train a lightweight 3D CNN to predict the same attributes from RGB volumes.

## Model Building

- Join appearance and motion attributes by `volume_id` into one model feature table.
- Train only on `split == training`, which is nominal Avenue video.
- Use region-specific greedy exemplar selection as the first anomaly model.
- Score test volumes by nearest exemplar distance within the same spatial region.
- Keep `avenue.mat` ground truth for evaluation after scoring, not for training.

## Evaluation Notes

- Current evaluation outputs frame AUC from max-projected anomaly scores and spatial-mask AUC/AP using Avenue `volLabel` masks.
- Further improvement: implement RBDC-style connected-component overlap from predicted spatial score maps and GT masks.
- Further ablation/improvement: approximate TBDC by linking GT mask connected components across time, unless official track IDs are found.

## Future Real-Time / Server-Streaming Work

- Current stride-10 implementation is an offline research pipeline, not a real-time runtime.
- Measured profile on the current workstation shows feature extraction is the bottleneck:
  - appearance extraction uses ResNet18 on GPU
  - Farneback optical flow, exemplar scoring, projection, and evaluation run on CPU
  - exemplar scoring is relatively fast once features exist
- Current unoptimized end-to-end estimate is only a few frame-equivalent FPS, so it is suitable for batch/offline analysis and visual validation, not yet live camera deployment.
- Main reason: the current volume-first scripts recompute overlapping crops and overlapping optical-flow pairs many times.
- Future server baseline should be frame-first and cached:
  - ingest camera frames and store raw video/frame chunks
  - compute full-frame optical flow once per adjacent frame pair
  - compute/crop/batch appearance features once per frame-region where possible
  - maintain a rolling 10-frame feature buffer per region
  - score only the new temporal window as frames arrive
  - emit frame heatmaps, anomaly events, and saved clips around peaks
- Optimization targets:
  - cache per-frame/per-region appearance embeddings before temporal pooling
  - cache per-frame-pair optical flow before region pooling
  - batch appearance crops aggressively on GPU
  - vectorize region exemplar scoring
  - use lower-resolution or ROI-only processing for multi-camera server deployment
- Goal: make the method a practical real-time-ish server baseline for Avenue-like resolution before discussing edge/chip deployment.
