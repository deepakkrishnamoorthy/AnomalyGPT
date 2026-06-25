# Exemplar + Volume LEJEPA Fusion, Stride 10

This experiment fuses two complementary anomaly signals:

- Exemplar score: region-specific distance from normal memory using the lightweight EVAL-style appearance and optical-flow features.
- Volume LEJEPA score: future-prediction latent error from frames 1-5 to frames 6-10.

The fusion script does not use labels. It robust-normalizes both per-volume scores with median and q95 scaling, then computes:

```text
fused_score = exemplar_weight * exemplar_norm + (1 - exemplar_weight) * lejepa_norm
```

Initial run:

```text
exemplar_weight = 0.75
lejepa_weight = 0.25
matched volumes = 69,435
```

Results:

| Method | Frame AUC | Frame AP | Spatial Frame AUC | Spatial Frame AP | Sampled Pixel AUC | Sampled Pixel AP |
|---|---:|---:|---:|---:|---:|---:|
| Exemplar stride-10 baseline | 0.8115 | 0.6356 | 0.8211 | 0.6502 | 0.9559 | 0.4861 |
| Volume LEJEPA Route B-lite | 0.7482 | 0.4851 | 0.7560 | 0.4855 | 0.8970 | 0.1830 |
| Fusion w=0.75 exemplar | 0.8342 | 0.6407 | 0.8456 | 0.6541 | 0.9504 | 0.4213 |

Interpretation:

The fused score improves temporal ranking over the stride-10 exemplar baseline, which suggests LEJEPA is adding a useful future-prediction signal. Pixel localization is still slightly worse than the pure exemplar baseline, likely because LEJEPA scores are latent prediction errors and do not preserve the same explicit spatial/motion decomposition.

