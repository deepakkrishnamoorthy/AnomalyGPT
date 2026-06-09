# Avenue Instrument Panel Feature EDA

## Summary

- Total clips: 2811
- Training clips: 949
- Testing clips: 1862
- Positive clips: 635
- Numeric feature columns: 95

## Feature Groups

- Object class bars: currently placeholder columns because no object classifier is integrated yet.
- Stationary/moving panel: aggregated from pseudo-track stationary fractions.
- Direction histogram: 12 bins weighted by pseudo-track length.
- Speed rays: mean pseudo-track speed per direction bin.
- Physics stats: velocity, acceleration, curvature, flow deviation, velocity ratio, track length.

## Selected Feature Means

| Feature | Train Mean | Test Mean | Positive Mean |
|---|---:|---:|---:|
| track_count | 18.1212 | 20.6450 | 25.2016 |
| candidate_track_fraction | 0.0000 | 0.2213 | 0.6489 |
| velocity_mean | 250.2369 | 272.2911 | 296.7043 |
| max_velocity_max | 1234.9808 | 1272.5182 | 1318.7011 |
| acceleration_mean | 6150.6085 | 6477.4890 | 6928.2159 |
| curvature_mean | 68.6413 | 70.3769 | 73.2204 |
| flow_deviation_mean | 32.6184 | 31.8594 | 33.5831 |
| stationary_panel_fraction | 0.0172 | 0.0152 | 0.0021 |
| moving_panel_fraction | 0.9828 | 0.9848 | 0.9979 |
| crowd_mean_speed_px_s | 250.2369 | 272.2911 | 296.7043 |

## Autoencoder Usage

Use `avenue_instrument_panel_train.csv` for normal-only training.
Use `avenue_instrument_panel_test.csv` for anomaly scoring and AUC evaluation.

Recommended first feature set: drop metadata and labels, then standardize the numeric instrument-panel columns.
