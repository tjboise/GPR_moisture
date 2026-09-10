import openpyxl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert

# ── Load data ──────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook('../data/GPR measurement data in field.xlsx')

CONDITIONS = {
    '2in_sand': {
        'gpr_sheet': '2 in sand gpr data',
        'moist_sheet': '2 in sand moisture data',
        'pipe_depth_m': 0.40,
        'label': '2" pipe, Sand (pipe top @ 40 cm)',
        'depths_cm': {'S': 0, 'T': 8, 'M': 22, 'B': 35},
    },
    '4in_sand': {
        'gpr_sheet': '4in sand gpr data',
        'moist_sheet': '4in sand mosture data',
        'pipe_depth_m': 0.35,
        'label': '4" pipe, Sand (pipe top @ 35 cm)',
        'depths_cm': {'S': 0, 'T': 8, 'M': 22, 'B': 35},
    },
    '4in_clay': {
        'gpr_sheet': '4in clay gpr data',
        'moist_sheet': '4in clay moisture data',
        'pipe_depth_m': 0.30,
        'label': '4" pipe, Clay (pipe top @ 30 cm)',
        'depths_cm': {'S': 0, 'T': 4, 'M': 17, 'B': 30},
    },
}

def load_condition(gpr_sname, moist_sname):
    gpr_ws = wb[gpr_sname]
    moist_ws = wb[moist_sname]
    gpr_rows = list(gpr_ws.iter_rows(values_only=True))
    moist_rows = list(moist_ws.iter_rows(values_only=True))

    time_ns = np.array([r[0] for r in gpr_rows[1:]], dtype=float)
    n_traces = gpr_ws.max_column - 1

    traces = np.zeros((len(time_ns), n_traces))
    for col in range(n_traces):
        for row_i, r in enumerate(gpr_rows[1:]):
            v = r[col + 1]
            traces[row_i, col] = v if v is not None else 0

    # Moisture: skip header row, skip avg/empty rows (where label is None)
    moisture = {}
    for row_i in range(1, len(moist_rows)):
        label = moist_rows[row_i][0]
        if label is None:
            continue
        vals = []
        for col in range(1, n_traces + 1):
            v = moist_rows[row_i][col] if col < len(moist_rows[row_i]) else None
            vals.append(float(v) if v is not None else np.nan)
        moisture[str(label)] = np.array(vals)

    return time_ns, traces, moisture, n_traces

data = {}
for key, cfg in CONDITIONS.items():
    t, tr, mo, n = load_condition(cfg['gpr_sheet'], cfg['moist_sheet'])
    # Normalize traces
    tr_norm = tr / np.max(np.abs(tr))
    # Envelope via Hilbert
    envelope = np.abs(hilbert(tr, axis=0))
    data[key] = {'time': t, 'traces': tr, 'traces_norm': tr_norm,
                 'envelope': envelope, 'moisture': mo, 'n': n}
    print(f"{key}: {n} traces, time 0–{t[-1]:.1f} ns, "
          f"moisture layers: {list(mo.keys())}")

# ── Figure 1: B-scan for each condition ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('GPR B-scan — Raw Signal Amplitude', fontsize=14, fontweight='bold')

for ax, (key, cfg) in zip(axes, CONDITIONS.items()):
    d = data[key]
    tr = d['traces']
    t = d['time']
    n = d['n']
    vmax = np.percentile(np.abs(tr), 95)
    im = ax.imshow(tr, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[1, n, t[-1], t[0]])
    ax.set_title(cfg['label'], fontsize=10)
    ax.set_xlabel('Trace No.')
    ax.set_ylabel('Two-way travel time (ns)')
    plt.colorbar(im, ax=ax, label='Amplitude')

plt.tight_layout()
plt.savefig('fig1_bscan.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_bscan.png')

# ── Figure 2: A-scan examples — dry vs wet (per condition) ────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('A-scan Comparison: Dry vs Wet (by bottom-layer moisture)', fontsize=13, fontweight='bold')

for ax, (key, cfg) in zip(axes, CONDITIONS.items()):
    d = data[key]
    mo = d['moisture']
    # Use B-layer moisture to pick dry/wet traces
    b_key = [k for k in mo if 'B' in k or 'B' in k.upper()][0]
    b_mo = mo[b_key]
    valid = ~np.isnan(b_mo)
    idx_valid = np.where(valid)[0]
    dry_idx = idx_valid[np.argmin(b_mo[valid])]
    wet_idx = idx_valid[np.argmax(b_mo[valid])]

    ax.plot(d['traces'][:, dry_idx] / 1e5, d['time'],
            color='#e8913a', lw=1.5, label=f'Dry (B={b_mo[dry_idx]:.1f}%)')
    ax.plot(d['traces'][:, wet_idx] / 1e5, d['time'],
            color='#3a7de8', lw=1.5, label=f'Wet (B={b_mo[wet_idx]:.1f}%)')
    ax.invert_yaxis()
    ax.set_title(cfg['label'], fontsize=9)
    ax.set_xlabel('Amplitude (×10⁵)')
    ax.set_ylabel('Two-way travel time (ns)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig('fig2_ascan_dry_vs_wet.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_ascan_dry_vs_wet.png')

# ── Figure 3: GPR features vs Moisture (scatter) ─────────────────────────
# Feature: peak-to-peak amplitude in full window, and in first half (0-12 ns)
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('GPR Peak-to-Peak Amplitude vs Moisture at Each Depth Layer', fontsize=13, fontweight='bold')

for row_i, (key, cfg) in enumerate(CONDITIONS.items()):
    d = data[key]
    tr = d['traces']
    t = d['time']
    mo = d['moisture']

    # Feature: peak-to-peak in 0–15 ns window
    t_mask = t <= 15.0
    feature = np.ptp(tr[t_mask, :], axis=0)  # peak-to-peak per trace

    layer_keys = [k for k in mo.keys()]
    colors = ['#e8913a', '#3a7de8', '#2ec4b6', '#e63946']

    for col_i, lk in enumerate(layer_keys[:4]):
        ax = axes[row_i, col_i]
        mo_vals = mo[lk]
        valid = ~np.isnan(mo_vals)
        ax.scatter(mo_vals[valid], feature[valid] / 1e6,
                   color=colors[col_i], alpha=0.7, s=40, edgecolors='white', lw=0.5)

        # Pearson r
        r = np.corrcoef(mo_vals[valid], feature[valid])[0, 1]

        # Fit line
        p = np.polyfit(mo_vals[valid], feature[valid][valid] if False else feature[valid], 1)
        x_line = np.linspace(mo_vals[valid].min(), mo_vals[valid].max(), 50)
        ax.plot(x_line, np.polyval(p, x_line) / 1e6, 'k--', lw=1, alpha=0.5)

        ax.set_title(f'{cfg["label"].split(",")[0]}\n{lk}  (r={r:.2f})', fontsize=8)
        ax.set_xlabel('Moisture (%)', fontsize=8)
        ax.set_ylabel('GPR Pk-Pk (×10⁶)', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_scatter_pkpk_vs_moisture.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_scatter_pkpk_vs_moisture.png')

# ── Figure 4: Envelope amplitude at pipe arrival time ─────────────────────
# Estimate pipe two-way travel time (rough: assume v~0.1 m/ns in moist soil)
# Use time window around pipe depth
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('GPR Envelope Energy (near pipe depth, 8–18 ns) vs Moisture', fontsize=13, fontweight='bold')

for row_i, (key, cfg) in enumerate(CONDITIONS.items()):
    d = data[key]
    env = d['envelope']
    t = d['time']
    mo = d['moisture']

    # Window around pipe reflection (rough: 8–18 ns covers 0.4–0.9m two-way at v=0.1m/ns)
    t_mask = (t >= 8) & (t <= 18)
    feature = np.mean(env[t_mask, :], axis=0)  # mean envelope energy in window

    layer_keys = list(mo.keys())
    colors = ['#e8913a', '#3a7de8', '#2ec4b6', '#e63946']

    for col_i, lk in enumerate(layer_keys[:4]):
        ax = axes[row_i, col_i]
        mo_vals = mo[lk]
        valid = ~np.isnan(mo_vals)
        ax.scatter(mo_vals[valid], feature[valid] / 1e5,
                   color=colors[col_i], alpha=0.7, s=40, edgecolors='white', lw=0.5)
        r = np.corrcoef(mo_vals[valid], feature[valid])[0, 1]
        p = np.polyfit(mo_vals[valid], feature[valid], 1)
        x_line = np.linspace(mo_vals[valid].min(), mo_vals[valid].max(), 50)
        ax.plot(x_line, np.polyval(p, x_line) / 1e5, 'k--', lw=1, alpha=0.5)
        ax.set_title(f'{cfg["label"].split(",")[0]}\n{lk}  (r={r:.2f})', fontsize=8)
        ax.set_xlabel('Moisture (%)', fontsize=8)
        ax.set_ylabel('Envelope energy (×10⁵)', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_envelope_vs_moisture.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_envelope_vs_moisture.png')

print('\nAll figures saved.')
