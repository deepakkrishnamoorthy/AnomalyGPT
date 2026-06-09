# Avenue Clip Dataset Validation

## Windowing

- Clip length: 32
- Training stride: 16
- Testing stride: 8

## Summary

- Total clips: 2811
- Training clips: 949
- Testing clips: 1862
- Positive clips: 635
- Positive testing clips: 635
- Total clip-frame observations: 89952
- Positive frame labels inside clips: 15323

Because clips overlap, positive frame labels inside clips are expected to exceed the unique anomalous frame count in `avenue_eda.md`.

## Per-Video Clip Counts

| Split | Video | Clips | Positive Clips | Mean Overlap |
|---|---:|---:|---:|---:|
| testing | 01 | 177 | 75 | 0.3185 |
| testing | 02 | 149 | 28 | 0.1166 |
| testing | 03 | 113 | 19 | 0.0962 |
| testing | 04 | 116 | 19 | 0.1002 |
| testing | 05 | 123 | 44 | 0.3232 |
| testing | 06 | 158 | 62 | 0.3426 |
| testing | 07 | 73 | 19 | 0.1751 |
| testing | 08 | 2 | 2 | 0.3125 |
| testing | 09 | 144 | 64 | 0.2862 |
| testing | 10 | 103 | 35 | 0.1975 |
| testing | 11 | 56 | 30 | 0.4062 |
| testing | 12 | 156 | 42 | 0.1995 |
| testing | 13 | 66 | 17 | 0.1534 |
| testing | 14 | 61 | 15 | 0.1414 |
| testing | 15 | 123 | 15 | 0.0915 |
| testing | 16 | 90 | 15 | 0.1347 |
| testing | 17 | 51 | 49 | 0.8689 |
| testing | 18 | 34 | 34 | 0.9559 |
| testing | 19 | 28 | 18 | 0.5625 |
| testing | 20 | 32 | 26 | 0.6016 |
| testing | 21 | 7 | 7 | 0.8482 |
| training | 01 | 85 | 0 | 0.0000 |
| training | 02 | 94 | 0 | 0.0000 |
| training | 03 | 92 | 0 | 0.0000 |
| training | 04 | 94 | 0 | 0.0000 |
| training | 05 | 50 | 0 | 0.0000 |
| training | 06 | 94 | 0 | 0.0000 |
| training | 07 | 68 | 0 | 0.0000 |
| training | 08 | 63 | 0 | 0.0000 |
| training | 09 | 86 | 0 | 0.0000 |
| training | 10 | 76 | 0 | 0.0000 |
| training | 11 | 48 | 0 | 0.0000 |
| training | 12 | 9 | 0 | 0.0000 |
| training | 13 | 22 | 0 | 0.0000 |
| training | 14 | 31 | 0 | 0.0000 |
| training | 15 | 22 | 0 | 0.0000 |
| training | 16 | 15 | 0 | 0.0000 |

## Validation Warnings

- None. Clip labels, frame labels, and video coverage checks passed.
