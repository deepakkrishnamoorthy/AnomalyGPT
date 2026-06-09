# Physics-Grounded Dataset Pipeline

This pipeline converts frame-level video anomaly datasets into the AnomalyGPT format:

```text
datasets/anomalygpt/
  avenue_testing_01_clip_000000/
    frames/
    masks/
    tracks.json
    physics.json
    qa.json
    meta.json
```

The first target is Avenue because it is already available locally and has verified frame-level anomaly intervals in `avenue.mat`.

## Stage 0: Raw Dataset Audit

Inputs:

- `datasets/avenue/training/frames/*`
- `datasets/avenue/testing/frames/*`
- `datasets/avenue/avenue.mat`

Outputs:

- `manifests/avenue_manifest.json`
- `reports/avenue_eda.md`

Current verified facts:

- 16 training videos
- 21 testing videos
- 30,652 total frames
- 47 test anomaly intervals
- 3,867 anomalous test frames
- annotations are frame-level intervals, not pixel masks

## Stage 1: Clip Windowing

Convert every video into fixed-length clips.

Recommended first settings:

```text
clip_len = 32 frames
stride_train = 16
stride_test = 8
```

Each clip gets:

- `split`: training or testing
- `video_id`
- `clip_id`
- `start_frame`
- `end_frame`
- `frame_paths`
- `frame_labels`
- `clip_label`
- `anomaly_overlap`

Frame labels are derived from the Avenue intervals.

Clip label rule:

```text
clip_label = 1 if any frame in the clip is anomalous else 0
```

For training:

- Avenue training clips are normal-only.
- Testing clips can be used for validation/evaluation and later supervised fine-tuning.

## Stage 2: Frame Materialization

For each clip, either symlink/copy frames into:

```text
frames/
  000000.jpg
  000001.jpg
  ...
```

Recommended first pass:

- Do not duplicate raw frames.
- Store relative paths in `meta.json`.
- Materialize frames only when a training loader requires local clip folders.

## Stage 3: Object Detection And Tracking

Run a detector/tracker over each clip.

Recommended tracker:

- ByteTrack for first pass
- BoT-SORT if camera motion or identity switches become a problem

Output:

```json
{
  "video_id": "01",
  "clip_id": "avenue_testing_01_clip_000000",
  "fps": 25.0,
  "tracks": [
    {
      "track_id": 4,
      "category": "person",
      "boxes": [
        {"frame": 1, "bbox": [120, 80, 164, 180], "score": 0.91},
        {"frame": 2, "bbox": [123, 80, 167, 180], "score": 0.93}
      ]
    }
  ]
}
```

Use 1-based frame indices inside clip-local metadata so they match annotation intervals cleanly.

## Stage 4: Physics Feature Extraction

For every track, compute:

- mean velocity
- max velocity
- acceleration
- direction
- curvature
- crowd-flow direction
- flow deviation
- stationary duration
- track length

Output:

```json
{
  "clip_id": "avenue_testing_01_clip_000000",
  "features": [
    {
      "track_id": 4,
      "velocity": 5.2,
      "acceleration": 1.3,
      "direction": 172.0,
      "trajectory_score": 0.84,
      "flow_deviation": 0.91,
      "is_candidate_anomaly": true
    }
  ],
  "crowd_flow": {
    "mean_direction": 18.0,
    "mean_speed": 1.2
  }
}
```

Candidate anomaly heuristics:

- velocity ratio above normal pedestrian flow
- high acceleration spike
- direction deviation above 90 degrees
- strong curvature or erratic path
- long loitering/stationary behavior in abnormal region

## Stage 5: Weak Mask Generation

Avenue annotations are frame-level, so pixel masks must be generated or obtained separately.

First weak-mask strategy:

1. Select anomalous clips using frame labels.
2. Pick candidate anomaly tracks using physics features.
3. Convert candidate boxes into coarse masks.
4. Optionally refine boxes with SAM/SAM2.

Output:

```text
masks/
  000000.png
  000001.png
```

Mask meaning:

```text
0 = background/normal
1 = anomalous region
255 = ignore/unknown
```

For normal clips, masks can be empty or omitted.

## Stage 6: QA And Explanation Generation

Generate `qa.json` for each clip.

Normal clip:

```json
{
  "question": "What anomaly is present?",
  "answer": "<assessment>No anomaly is present. Pedestrians follow normal scene motion.</assessment><physics>velocity and direction are consistent with normal crowd flow.</physics>",
  "target_track_ids": [],
  "clip_label": 0
}
```

Anomalous clip:

```json
{
  "question": "Explain why the highlighted behavior is abnormal.",
  "answer": "<assessment>The highlighted person is anomalous because their motion differs from the surrounding pedestrian flow.</assessment><physics>velocity_ratio=3.8; direction_deviation=142 degrees; flow_deviation=0.79</physics><SEG>",
  "target_track_ids": [4],
  "clip_label": 1
}
```

The explanation should be generated from physics features, not hand-written blindly.

## Stage 7: AnomalyGPT Sample Manifest

Create a training manifest:

```text
manifests/anomalygpt_avenue_clips.jsonl
```

One line per clip:

```json
{
  "clip_id": "avenue_testing_01_clip_000000",
  "split": "testing",
  "dataset": "avenue",
  "frames": ["datasets/avenue/testing/frames/01/0000.jpg"],
  "meta": "datasets/anomalygpt/avenue_testing_01_clip_000000/meta.json",
  "tracks": "datasets/anomalygpt/avenue_testing_01_clip_000000/tracks.json",
  "physics": "datasets/anomalygpt/avenue_testing_01_clip_000000/physics.json",
  "qa": "datasets/anomalygpt/avenue_testing_01_clip_000000/qa.json"
}
```

This keeps large frame data out of git while making the dataset loader deterministic.

## Stage 8: Dataset Loader Integration

Create an `AnomalyGPTDataset` modeled after WalkGPT's `PAVEDataset`.

The loader should emit the same core fields WalkGPT expects:

- image or selected keyframe tensor
- clip metadata
- conversations
- masks
- labels
- resize info
- questions payload
- sampled target classes or track ids

First compatibility mode:

- Use the middle frame or highest-anomaly frame as the SAM image.
- Keep the clip frames available for physics and future temporal encoding.

Second mode:

- Add temporal frame encoding and fuse clip-level physics features.

## Stage 9: Evaluation

Available immediately:

- Frame AUC
- clip accuracy/F1
- temporal interval IoU after thresholding frame scores

Available after weak or real masks:

- pixel AUC
- IoU
- Dice

Available after explanation generation:

- explanation consistency with measured physics
- grounding score against selected tracks/masks

## Implementation Order

1. Build clip manifest from `manifests/avenue_manifest.json`.
2. Generate frame labels and clip labels.
3. Create `qa.json` with normal/anomaly assessment text.
4. Add simple temporal metrics for frame labels and interval IoU.
5. Add tracking.
6. Compute physics features.
7. Generate weak masks.
8. Implement `AnomalyGPTDataset`.
9. Plug dataset into WalkGPT training/evaluation.
