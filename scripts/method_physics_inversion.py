"""
Physics-based inversion from single A-scan:
  1. Pick pipe reflection arrival time via Hilbert envelope peak
  2. v = 2 * pipe_depth / t_twoway
  3. eps_r = (c / v)^2
  4. Topp equation -> volumetric water content
  5. Compare with TDR ground truth
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import pearsonr
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT  = ROOT / 'results'

# ── constants ──────────────────────────────────────────────────────────────
C  = 3e8          # m/s speed of light
DT = 0.099609e-9  # s per sample

# pipe TOP depth by condition label
PIPE_DEPTH = {0: 0.40, 1: 0.35, 2: 0.30}
COND_NAMES = {0: 'Sand 2"', 1: 'Sand 4"', 2: 'Clay 4"'}

# physically plausible ε_r range for these soils
EPS_MIN = 5    # very dry sand lower bound
EPS_MAX = 35   # very wet clay upper bound


# ── Topp equation ──────────────────────────────────────────────────────────
def topp(eps_r):
    """eps_r -> volumetric water content (fraction)"""
    return -5.3e-2 + 2.92e-2*eps_r - 5.5e-4*eps_r**2 + 4.3e-6*eps_r**3


def find_first_break(ascan, threshold_frac=0.05):
    """First sample where |signal| exceeds threshold_frac * peak (same as data_prep.py)."""
    peak = np.max(np.abs(ascan))
    if peak == 0:
        return 0
    above = np.where(np.abs(ascan) > threshold_frac * peak)[0]
    return int(above[0]) if len(above) > 0 else 0


def pick_arrival(ascan, pipe_depth):
    """
    Find pipe reflection arrival time.
    - t0 = first break (soil surface entry) found per trace
    - Search window measured FROM t0, constrained by physically expected ε_r range
    - Two-way travel time = (peak_sample - t0) * DT  →  v  →  ε_r
    """
    t0 = find_first_break(ascan)
    t_min = 2 * pipe_depth * np.sqrt(EPS_MIN) / C
    t_max = 2 * pipe_depth * np.sqrt(EPS_MAX) / C
    s_min = t0 + int(t_min / DT)
    s_max = t0 + int(t_max / DT)
    s_max = min(s_max, len(ascan) - 1)
    env = np.abs(hilbert(ascan))
    region = env[s_min:s_max]
    if len(region) == 0:
        return s_min, t0
    return int(np.argmax(region)) + s_min, t0


# ── load data ──────────────────────────────────────────────────────────────
X    = np.load(DATA / 'X_norm.npy')          # (135, 256)
y    = np.load(DATA / 'y_moisture.npy')       # (135, 4)  S/T/M/B  in %
cond = np.load(DATA / 'cond_labels.npy')      # (135,)
time = np.load(DATA / 'time_ns.npy')          # (256,)  ns

n = len(X)

# ── inversion loop ─────────────────────────────────────────────────────────
eps_r_pred   = np.zeros(n)
theta_pred   = np.zeros(n)
arrival_samp = np.zeros(n, dtype=int)
t0_samp      = np.zeros(n, dtype=int)

for i in range(n):
    d         = PIPE_DEPTH[cond[i]]
    samp, t0  = pick_arrival(X[i], d)
    t_soil    = (samp - t0) * DT        # two-way travel time in soil only
    if t_soil <= 0:
        t_soil = DT                     # guard against zero division
    v         = 2 * d / t_soil
    eps       = (C / v) ** 2
    eps_r_pred[i]   = eps
    theta_pred[i]   = topp(eps) * 100
    arrival_samp[i] = samp
    t0_samp[i]      = t0

# TDR reference: average of S and T (0 and 8 cm) as proxy for surface-to-pipe
y_ref = y[:, :2].mean(axis=1)   # mean of S+T layers

# ── Figure 1: example A-scans with picked arrival ──────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
fig.suptitle('A-scan with picked pipe reflection (red line = arrival, green = envelope)',
             fontsize=11, fontweight='bold')

for ci, (c_label, ax_row) in enumerate(zip([0, 1, 2], axes)):
    idx = np.where(cond == ci)[0][:3]
    for j, (idx_s, ax) in enumerate(zip(idx, ax_row)):
        env = np.abs(hilbert(X[idx_s]))
        ax.plot(time, X[idx_s], color='steelblue', lw=1.2, label='A-scan')
        ax.plot(time, env,      color='limegreen',  lw=1.0, alpha=0.7, label='envelope')
        ax.axvline(time[t0_samp[idx_s]],      color='purple', lw=1.2, linestyle=':',
                   label=f't0={time[t0_samp[idx_s]]:.1f} ns (first break)')
        ax.axvline(time[arrival_samp[idx_s]], color='red',    lw=1.5, linestyle='--',
                   label=f'pipe={time[arrival_samp[idx_s]]:.1f} ns')
        ax.set_title(f'{COND_NAMES[ci]} | sample #{idx_s}\n'
                     f'ε_r={eps_r_pred[idx_s]:.1f}  θ_Topp={theta_pred[idx_s]:.1f}%  '
                     f'TDR_avg={y_ref[idx_s]:.1f}%  t0_samp={t0_samp[idx_s]}', fontsize=7)
        ax.set_ylim(-1.2, 1.5)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel('Time (ns)', fontsize=8)
plt.tight_layout()
fig.savefig(OUT / 'fig_physics_arrival.png', dpi=150, bbox_inches='tight')
print('Saved fig_physics_arrival.png')
plt.close()

# ── Figure 2: Topp-predicted vs TDR reference scatter ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle('Physics inversion (Topp) vs TDR moisture\n'
             'Topp moisture = travel-time estimate for surface-to-pipe layer\n'
             'TDR reference = mean of S+T layers (0 & 8 cm)',
             fontsize=10, fontweight='bold')

colors = ['steelblue', 'tomato', 'seagreen']
for ci in range(3):
    ax  = axes[ci]
    idx = cond == ci
    x_  = theta_pred[idx]
    y_  = y_ref[idx]
    r, _ = pearsonr(x_, y_)
    residuals = x_ - y_
    rmse = np.sqrt(np.mean(residuals**2))

    ax.scatter(y_, x_, color=colors[ci], alpha=0.7, s=40, edgecolors='k', lw=0.3)
    lims = [min(y_.min(), x_.min()) - 2, max(y_.max(), x_.max()) + 2]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('TDR mean (S+T) %', fontsize=9)
    ax.set_ylabel('Topp predicted %', fontsize=9)
    ax.set_title(f'{COND_NAMES[ci]}\nr={r:.2f}  RMSE={rmse:.1f}%', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / 'fig_physics_topp_scatter.png', dpi=150, bbox_inches='tight')
print('Saved fig_physics_topp_scatter.png')
plt.close()

# ── Figure 3: ε_r distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for ci in range(3):
    idx = cond == ci
    ax.hist(eps_r_pred[idx], bins=15, alpha=0.6, label=COND_NAMES[ci], color=colors[ci])
ax.set_xlabel('Estimated ε_r', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Distribution of estimated relative permittivity by condition', fontsize=10)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'fig_physics_epsr_dist.png', dpi=150, bbox_inches='tight')
print('Saved fig_physics_epsr_dist.png')
plt.close()

# ── print summary ──────────────────────────────────────────────────────────
print('\n=== Physics Inversion Summary ===')
print(f'{"Cond":<12} {"ε_r mean":>10} {"ε_r std":>8} {"Topp mean%":>12} {"TDR mean%":>10} {"r":>6} {"RMSE%":>7}')
for ci in range(3):
    idx = cond == ci
    r, _ = pearsonr(theta_pred[idx], y_ref[idx])
    rmse  = np.sqrt(np.mean((theta_pred[idx] - y_ref[idx])**2))
    print(f'{COND_NAMES[ci]:<12} {eps_r_pred[idx].mean():>10.2f} {eps_r_pred[idx].std():>8.2f} '
          f'{theta_pred[idx].mean():>12.1f} {y_ref[idx].mean():>10.1f} {r:>6.3f} {rmse:>7.2f}')
