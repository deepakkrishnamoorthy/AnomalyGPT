# Stride-1 Avenue Experiment

This experiment keeps all stride-1 artifacts separate from the existing stride-10 baseline.

## Why This Exists

The current baseline uses 10-frame volumes with temporal stride `10`. This folder is for the denser EVAL-style temporal stride `1` ablation.

Stride `1` means every possible 10-frame window is used:

- higher temporal coverage
- better chance of localizing short anomalies
- roughly 10x more volume rows than stride `10`
- much heavier appearance, motion, scoring, and storage cost

## Artifact Layout

```text
experiments/stride1/
  manifests/
  data/
  features/
  models/
  outputs/
  reports/
```

## Run From

Run every command from:

```powershell
cd D:\Deepak\VAD-Folder\Freshstart
```

## Step 1: Build Stride-1 Manifest

```powershell
python scripts\03_build_eval_volume_manifest.py --depth 10 --region-size 128 --temporal-stride 1 --out experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl
```

This creates dense 10-frame spatial volumes with one-frame temporal stride. It only writes metadata, not image arrays.

## Step 2: Smoke-Test Volume Export

```powershell
python scripts\05_export_eval_volumes.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out-root experiments\stride1\data\avenue_eval10_stride1_volumes --index-out experiments\stride1\manifests\avenue_eval10_stride1_saved_volumes_smoke_manifest.jsonl --limit 500
```

This saves only 500 `.npz` volumes and preview contact sheets so we can visually confirm the stride-1 extraction before committing to the full run.

## Step 3: Full Volume Export

```powershell
python scripts\05_export_eval_volumes.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out-root experiments\stride1\data\avenue_eval10_stride1_volumes --index-out experiments\stride1\manifests\avenue_eval10_stride1_saved_volumes_manifest.jsonl --no-preview --skip-existing
```

This saves full `.npz` volumes only. Preview images for every stride-1 volume are intentionally skipped because they can consume a lot of disk space.

Important: the appearance and motion scripts read frames directly from `data/avenue_frames`, so full `.npz` export is useful for validation/debugging but is not required for feature extraction.

## Step 4: Smoke-Test Motion Features

```powershell
python scripts\08_extract_eval_motion_attributes.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out-csv experiments\stride1\features\avenue_eval10_stride1_motion_attributes_smoke.csv --out-jsonl experiments\stride1\manifests\avenue_eval10_stride1_motion_attributes_smoke.jsonl --limit 500
```

## Step 5: Full Motion Features

```powershell
python scripts\08_extract_eval_motion_attributes.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out-csv experiments\stride1\features\avenue_eval10_stride1_motion_attributes.csv --out-jsonl experiments\stride1\manifests\avenue_eval10_stride1_motion_attributes.jsonl
```

This computes direct optical-flow attributes for every stride-1 volume.

## Step 6: Smoke-Test Appearance Features

```powershell
python scripts\06_extract_appearance_features.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out experiments\stride1\features\avenue_eval10_stride1_appearance_resnet18_smoke.csv --limit 500 --device cuda
```

If CUDA is not available, replace `--device cuda` with `--device cpu`.

## Step 7: Full Appearance Features

```powershell
python scripts\06_extract_appearance_features.py --manifest experiments\stride1\manifests\avenue_eval10_stride1_volume_manifest.jsonl --out experiments\stride1\features\avenue_eval10_stride1_appearance_resnet18.csv --device cuda
```

This is expected to be the longest feature-extraction step because each volume crops 10 RGB frames and passes them through ResNet18.

## Later Modeling Commands

After motion and appearance finish, the downstream commands are:

```powershell
python scripts\09_build_model_feature_table.py --appearance experiments\stride1\features\avenue_eval10_stride1_appearance_resnet18.csv --motion experiments\stride1\features\avenue_eval10_stride1_motion_attributes.csv --out experiments\stride1\features\avenue_eval10_stride1_model_features.csv
```

```powershell
python scripts\10_train_region_exemplars.py --features experiments\stride1\features\avenue_eval10_stride1_model_features.csv --out experiments\stride1\models\avenue_eval10_stride1_region_exemplars.pkl
```

```powershell
python scripts\11_score_region_exemplars.py --features experiments\stride1\features\avenue_eval10_stride1_model_features.csv --model experiments\stride1\models\avenue_eval10_stride1_region_exemplars.pkl --out experiments\stride1\outputs\avenue_eval10_stride1_exemplar_scores.csv
```

```powershell
python scripts\13_build_frame_scores.py --scores experiments\stride1\outputs\avenue_eval10_stride1_exemplar_scores.csv --out experiments\stride1\outputs\avenue_eval10_stride1_frame_scores.csv --spatial-map-dir experiments\stride1\outputs\avenue_eval10_stride1_spatial_score_maps
```

```powershell
python scripts\14_evaluate_frame_auc.py --frame-scores experiments\stride1\outputs\avenue_eval10_stride1_frame_scores.csv --out-summary experiments\stride1\outputs\avenue_eval10_stride1_frame_auc_summary.json --out-per-video experiments\stride1\outputs\avenue_eval10_stride1_frame_auc_per_video.csv
```

```powershell
python scripts\15_evaluate_spatial_mask_auc.py --spatial-map-dir experiments\stride1\outputs\avenue_eval10_stride1_spatial_score_maps --out-summary experiments\stride1\outputs\avenue_eval10_stride1_spatial_mask_auc_summary.json --out-per-video experiments\stride1\outputs\avenue_eval10_stride1_spatial_mask_auc_per_video.csv
```
