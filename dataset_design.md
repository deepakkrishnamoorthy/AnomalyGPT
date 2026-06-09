# AnomalyGPT Dataset Design

Each video clip becomes one AnomalyGPT sample directory.

```text
datasets/anomalygpt/
  video_id/
    frames/
    masks/
    tracks.json
    qa.json
```

## Frames

`frames/` contains ordered image frames for the clip, typically 16-32 frames.

Recommended naming:

```text
000001.jpg
000002.jpg
...
```

## Masks

`masks/` contains anomaly masks aligned with frames when available. For weakly annotated datasets, masks may be generated from tracked anomaly boxes, SAM2, or Grounded SAM.

Recommended naming:

```text
000001.png
000002.png
...
```

## Tracks

`tracks.json` stores object trajectories and tracker outputs.

```json
{
  "video_id": "example_video",
  "fps": 30.0,
  "tracks": [
    {
      "track_id": 4,
      "category": "person",
      "boxes": [
        {"frame": 1, "bbox": [120, 80, 164, 180]},
        {"frame": 2, "bbox": [123, 80, 167, 180]}
      ]
    }
  ]
}
```

## QA

`qa.json` stores the anomaly prompt, answer, physics measurements, and segmentation target references.

```json
{
  "video_id": "example_video",
  "question": "What anomaly is present?",
  "answer": "A cyclist is moving through a pedestrian-only area at high speed.",
  "physics": {
    "velocity": 5.1,
    "direction_deviation": 180.0
  },
  "target_track_ids": [4],
  "mask_frames": ["masks/000001.png"]
}
```
