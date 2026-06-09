# Avenue Dataset EDA

## Summary

- Training videos: 16
- Testing videos: 21
- Total frames: 30652
- Training frames: 15328
- Testing frames: 15324
- Test videos with anomalies: 21
- Ground-truth anomaly intervals: 47
- Ground-truth anomalous test frames: 3867
- Test anomaly frame fraction: 0.2523

## Annotation Format

`avenue.mat` contains one variable, `gt`, with 21 entries. Each entry stores inclusive 1-based frame intervals for the corresponding testing video.

Example interpretation:

```text
testing video 01: [78,120], [392,422], ...
```

These are frame-level labels. Pixel-level IoU/Dice requires separate masks or generated weak masks.

## Evaluation Notes

- Frame-level anomaly detection should use frame scores against the binary labels induced by these intervals.
- Frame AUC is the first reliable metric available from this annotation file.
- Interval IoU can compare predicted anomalous frame ranges with ground-truth ranges after thresholding frame scores.
- Pixel IoU / Dice can be added after we create or obtain anomaly masks.

Interval IoU definition:

```text
IoU = anomalous_frame_intersection / anomalous_frame_union
```

## Testing Videos

| Video | Frames | Intervals | Anomaly Frames | Fraction |
|---:|---:|---|---:|---:|
| 01 | 1439 | 78-120, 392-422, 503-666, 868-910, 932-1101 | 451 | 0.3134 |
| 02 | 1211 | 273-320, 724-764, 1051-1100 | 139 | 0.1148 |
| 03 | 923 | 295-340, 582-622 | 87 | 0.0943 |
| 04 | 947 | 380-428, 649-692 | 93 | 0.0982 |
| 05 | 1007 | 469-786 | 318 | 0.3158 |
| 06 | 1283 | 345-625, 856-1007 | 433 | 0.3375 |
| 07 | 605 | 423-494, 563-595 | 105 | 0.1736 |
| 08 | 36 | 21-30 | 10 | 0.2778 |
| 09 | 1175 | 136-183, 496-566, 741-755, 875-981, 1013-1044, 1104-1163 | 333 | 0.2834 |
| 10 | 841 | 571-607, 637-656, 678-713, 724-755, 783-818 | 161 | 0.1914 |
| 11 | 472 | 21-164, 308-346 | 183 | 0.3877 |
| 12 | 1271 | 539-617, 645-729, 759-843 | 249 | 0.1959 |
| 13 | 549 | 259-286, 458-510 | 81 | 0.1475 |
| 14 | 507 | 399-455, 485-500 | 73 | 0.1440 |
| 15 | 1001 | 498-587 | 90 | 0.0899 |
| 16 | 740 | 632-730 | 99 | 0.1338 |
| 17 | 426 | 21-56, 99-420 | 358 | 0.8404 |
| 18 | 294 | 21-285 | 265 | 0.9014 |
| 19 | 248 | 109-240 | 132 | 0.5323 |
| 20 | 273 | 65-144, 168-241 | 154 | 0.5641 |
| 21 | 76 | 14-66 | 53 | 0.6974 |

## Training Videos

| Video | Frames | Resolution |
|---:|---:|---|
| 01 | 1364 | 640x360 |
| 02 | 1511 | 640x360 |
| 03 | 1487 | 640x360 |
| 04 | 1511 | 640x360 |
| 05 | 815 | 640x360 |
| 06 | 1511 | 640x360 |
| 07 | 1099 | 640x360 |
| 08 | 1017 | 640x360 |
| 09 | 1391 | 640x360 |
| 10 | 1223 | 640x360 |
| 11 | 781 | 640x360 |
| 12 | 145 | 640x360 |
| 13 | 366 | 640x360 |
| 14 | 510 | 640x360 |
| 15 | 353 | 640x360 |
| 16 | 244 | 640x360 |

## Validation Warnings

- None. All annotated intervals fit within available testing frame counts.
