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
- **Sampling frequency:** fs = 1e9 / 0.099609 ≈ 10.04 GHz
- **Train / Test split:** 80 / 20 (stratified by condition)



## STFT Preprocessing

Short-Time Fourier Transform (STFT) converts each 1D A-scan into a 2D time-frequency spectrogram image, which is then fed into a 2D CNN. This captures how the frequency content of the GPR signal evolves over time — useful because moisture affects both the amplitude and the frequency-dependent attenuation of the EM wave.

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `fs` | 1e9 / 0.099609 ≈ 10.04 GHz | Sampling frequency |
| `nperseg` | 32 samples (~3.2 ns) | Sliding window length — smaller = better time resolution |
| `noverlap` | 28 samples (87.5%) | High overlap gives smooth time axis |
| `nfft` | 512 | Zero-padding for finer frequency grid |
| Frequency range | 0 – 3 GHz | Only physically meaningful range kept |
| Output image size | 153 × 57 (freq × time bins) | Input to 2D CNN |

Window choice trade-off: a shorter window (nperseg=32 vs the 56 used in the reference paper) gives better **time resolution** (~3.2 ns) at the cost of slightly coarser **frequency resolution**. This matters here because the pipe reflection is a brief event in time, and we want to resolve it precisely.

### STFT image examples

Low moisture samples (top row) vs high moisture samples (bottom row), sorted by B-layer moisture content:

![STFT sample images](fig_methodD_stft_samples.png)

Each image shows:
- **X-axis**: time bins (covering the full ~25 ns A-scan window)
- **Y-axis**: frequency bins (0 – 3 GHz)
- **Color**: signal energy (bright = high energy)

The bright cluster in the lower-left region corresponds to the main GPR pulse energy concentrated at low frequencies and early arrival time. Changes in moisture shift and attenuate this energy pattern, which the 2D CNN learns to map to moisture values.

---


## Reference

Cao, Q., Al-Qadi, I. L., & Abufares, L. (2022). Pavement Moisture Content Prediction: A Deep Residual Neural Network Approach for Analyzing Ground Penetrating Radar. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1–11.
