"""
Parse gprMax simulation data (1.2 GHz sheet).
For each of 125 cases:
  - Resample A-scan from 5194 → 256 samples (matches real data input size)
  - Normalize to [-1, 1] by peak absolute value
  - Compute moisture at S/T/M/B depths via layer ε_r + Topp equation

Outputs saved to data/sim/:
  X_sim.npy        (125, 256)  resampled + normalised A-scans
  y_sim.npy        (125, 4)    moisture [S, T, M, B] in %
  pipe_depth_sim.npy (125,)    pipe crown depth in m
"""

import numpy as np
import openpyxl
from scipy.signal import resample
import pathlib

ROOT   = pathlib.Path(__file__).resolve().parent.parent
EXCEL  = ROOT / 'data' / 'Summary_ GPR on layered soil.xlsx'
OUT    = ROOT / 'data' / 'sim'

N_OUT  = 256          # target sample count (matches real data)
SHEETS = ['A-scan data_0.4GHz', 'A-scan data_1.2GHz', 'A-scan data_1.5GHz']

# Depths for S/T/M/B moisture labels (metres)
LABEL_DEPTHS = {'S': 0.00, 'T': 0.08, 'M': 0.22, 'B': 0.35}


# ── Topp equation ──────────────────────────────────────────────────────────
def topp(eps_r):
    """ε_r → volumetric water content (fraction 0–1)."""
    theta = -5.3e-2 + 2.92e-2*eps_r - 5.5e-4*eps_r**2 + 4.3e-6*eps_r**3
    return float(np.clip(theta, 0, None))


def eps_at_depth(layers, depth_m):
    """
    Return ε_r at depth_m given a list of (top, bottom, eps_r) layer tuples.
    If depth exceeds all layer bottoms, use the deepest layer's ε_r.
    """
    for top, bot, eps in layers:
        if top <= depth_m < bot:
            return eps
    # depth is below all defined layers → use deepest layer
    return layers[-1][2]


# ── Load conditions ────────────────────────────────────────────────────────
print('Loading conditions...')
wb = openpyxl.load_workbook(EXCEL, data_only=True, read_only=True)

ws_cond = wb['Layered conditions']
rows_cond = list(ws_cond.iter_rows(min_row=2, values_only=True))

# Build per-case layer list and pipe depth
case_layers     = {}   # case_id → [(top, bottom, eps_r), ...]
case_pipe_depth = {}   # case_id → pipe crown depth (m)

for row in rows_cond:
    if row[0] is None:
        continue
    cid    = int(row[0])
    top    = float(row[2])   # Depth top (m)
    bot    = float(row[3])   # Depth bottom (m)
    eps_r  = float(row[7])   # Dielectric constant
    p_dep  = float(row[11])  # Pipe crown depth (m)

    if cid not in case_layers:
        case_layers[cid]     = []
        case_pipe_depth[cid] = p_dep
    case_layers[cid].append((top, bot, eps_r))

n_cases = len(case_layers)
print(f'  {n_cases} cases loaded')

# ── Compute moisture labels ────────────────────────────────────────────────
y_sim = np.zeros((n_cases, 4), dtype=np.float32)

for cid in sorted(case_layers.keys()):
    idx   = cid - 1           # 0-indexed
    lyrs  = case_layers[cid]
    for li, (lbl, depth) in enumerate(LABEL_DEPTHS.items()):
        eps   = eps_at_depth(lyrs, depth)
        theta = topp(eps) * 100.0   # convert to %
        y_sim[idx, li] = theta

print(f'  Moisture range: {y_sim.min():.1f}% ~ {y_sim.max():.1f}%')

# ── Load and resample A-scans (all 3 frequencies) ─────────────────────────
X_all_freq = []
for sheet_name in SHEETS:
    print(f'\nLoading A-scans from sheet "{sheet_name}"...')
    ws_ascan = wb[sheet_name]
    all_rows = list(ws_ascan.iter_rows(min_row=2, values_only=True))
    n_samples_raw = len(all_rows)

    X_raw = np.zeros((n_samples_raw, n_cases), dtype=np.float64)
    for ri, row in enumerate(all_rows):
        for ci in range(n_cases):
            v = row[ci + 1]
            X_raw[ri, ci] = float(v) if v is not None else 0.0

    X_freq = np.zeros((n_cases, N_OUT), dtype=np.float32)
    for ci in range(n_cases):
        trace = resample(X_raw[:, ci], N_OUT)
        peak  = np.max(np.abs(trace))
        if peak > 0:
            trace /= peak
        X_freq[ci] = trace.astype(np.float32)

    X_all_freq.append(X_freq)
    print(f'  Done: {X_freq.shape}')

# Stack all frequencies: each case appears 3 times (once per frequency)
# Labels and pipe depth are repeated accordingly
X_sim          = np.vstack(X_all_freq)                     # (375, 256)
y_sim_all      = np.tile(y_sim, (len(SHEETS), 1))          # (375, 4)
pipe_depth_arr = np.array(
    [case_pipe_depth[cid] for cid in sorted(case_layers.keys())],
    dtype=np.float32)
pipe_depth_all = np.tile(pipe_depth_arr, len(SHEETS))      # (375,)

# ── Save ───────────────────────────────────────────────────────────────────
np.save(OUT / 'X_sim.npy',          X_sim)
np.save(OUT / 'y_sim.npy',          y_sim_all)
np.save(OUT / 'pipe_depth_sim.npy', pipe_depth_all)

print(f'\nSaved to {OUT}:')
print(f'  X_sim.npy          {X_sim.shape}  range [{X_sim.min():.3f}, {X_sim.max():.3f}]')
print(f'  y_sim.npy          {y_sim_all.shape}')
print(f'  pipe_depth_sim.npy {pipe_depth_all.shape}')
print(f'\nMoisture per depth (sim):')
for li, lbl in enumerate(['S', 'T', 'M', 'B']):
    v = y_sim_all[:, li]
    print(f'  {lbl}: mean={v.mean():.1f}%  std={v.std():.1f}%  '
          f'range=[{v.min():.1f}, {v.max():.1f}]')
print(f'\nPipe crown depth values: {sorted(set(pipe_depth_all.tolist()))}')
