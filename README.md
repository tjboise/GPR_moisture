# GPR Moisture Prediction

Using Ground Penetrating Radar (GPR) A-scan signals to predict soil moisture content at multiple depths.

---

## Experiment Setup

A controlled box experiment with three conditions:

| Condition | Pipe diameter | Soil type | Pipe top depth |
|-----------|--------------|-----------|----------------|
| 2 in sand | 2" | Sand | 0.40 m |
| 4 in sand | 4" | Sand | 0.35 m |
| 4 in clay | 4" | Clay | 0.30 m |

The box was filled with soil and watered to different moisture levels. GPR A-scans were collected at each moisture condition, and soil moisture was measured at 4 depths simultaneously.

### Moisture measurement depths

| Layer | Sand depth | Clay depth |
|-------|-----------|------------|
| S (Surface) | ground surface | ground surface |
| T (Top) | 8 cm | ~4 cm |
| M (Middle) | 22 cm | ~17 cm |
| B (Bottom) | 35 cm | ~30 cm |

### Dataset summary

- **Total samples:** 112 A-scans (39 + 40 + 33 across three conditions)
- **A-scan length:** 256 time samples, dt = 0.099609 ns (~25 ns total window)
- **Sampling frequency:** ~10.04 GHz
- **Train / Test split:** 80 / 20 (stratified by condition)

---

## Data Files

| File | Description |
|------|-------------|
| `GPR measurement data in field.xlsx` | Raw GPR signals and moisture measurements (6 sheets) |
| `X_norm.npy` | Normalized, time-zero-aligned A-scans — shape (112, 256) |
| `y_moisture.npy` | Moisture labels — shape (112, 4), columns: S / T / M / B |
| `cond_labels.npy` | Condition index (0=2in_sand, 1=4in_sand, 2=4in_clay) |
| `idx_train.npy` | Training set indices (89 samples) |
| `idx_test.npy` | Test set indices (23 samples) |

---

## Analysis Plan

| # | Method | Script | Status |
|---|--------|--------|--------|
| 0 | Data preparation (load, align, split) | `data_prep.py` | Done |
| A | Hand-crafted features + Random Forest / SVR | `method_a_handcrafted.py` | Done |
| B | 1D CNN — raw A-scan → moisture | `method_b_1dcnn.py` | In progress |
| C | FFT spectrum features + ML | `method_c_fft.py` | Pending |
| D | STFT (tuned) + 2D CNN | `method_d_stft_cnn.py` | Pending |

---

## Hand-Crafted Features (Method A)

Extracted from each aligned A-scan:

1. Peak amplitude
2. Peak-to-peak amplitude
3. Signal energy
4. Envelope maximum
5. Envelope area (integral)
6. Time of envelope peak
7. Dominant frequency (FFT)
8. Spectral centroid
9. Energy ratio (first half / total)
10. RMS amplitude
11. Zero-crossing rate
12–14. Mean envelope amplitude in early / mid / late time windows

---

## Reference

Cao, Q., Al-Qadi, I. L., & Abufares, L. (2022). Pavement Moisture Content Prediction: A Deep Residual Neural Network Approach for Analyzing Ground Penetrating Radar. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1–11.
