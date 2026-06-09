# Avenue Motion Physics Summary

Motion pseudo-tracks and physics features were computed for every generated Avenue clip.

## Method

- Tracker: OpenCV MOG2 foreground segmentation plus contour centroid linking
- Track category: `motion_blob`
- This is a lightweight pseudo-tracking baseline, not ByteTrack or BoT-SORT.
- Ground truth remains the Avenue frame interval labels stored separately in `manifests/avenue_clip_ground_truth.json`.

## Parameters

- `min_area`: 80
- `max_area`: 60000
- `max_distance`: 55.0
- `max_missed`: 2
- `min_track_len`: 3

## Results

- Processed clips: 2811
- Clips with at least one track: 2771
- Total pseudo-tracks: 55638
- Candidate anomaly tracks: 10804
- Mean tracks per clip: 19.792956

## Outputs

Per-clip local files under `datasets/anomalygpt/avenue_clips/`:

- `meta.json`
- `qa.json`
- `tracks.json`
- `physics.json`
- `masks/`

Tracked project outputs:

- `manifests/anomalygpt_avenue_clips.jsonl`
- `manifests/avenue_clip_ground_truth.json`
- `reports/avenue_clip_dataset_validation.md`
- `reports/avenue_motion_physics_summary.json`

## Important Caveat

The pseudo-tracks are useful for bootstrapping physics features, but they are not final object tracks. For training the full physics-grounded model, the next upgrade should replace this tracker with ByteTrack or BoT-SORT using a person/object detector.
