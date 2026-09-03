"""
Summary comparison of all 4 methods.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAYERS = ['S', 'T', 'M', 'B']

# ── Collect results ────────────────────────────────────────────────────────
# Method A: best model per layer across RF / SVR / GB
resA_raw = np.load('results_methodA.npy', allow_pickle=True).item()
# Pick best R² per layer across the 3 models
r2_A  = np.max([[v['r2'][li]   for li in range(4)] for v in resA_raw.values()], axis=0)
rms_A = [resA_raw[max(resA_raw, key=lambda k: resA_raw[k]['r2'][li])]['rmse'][li]
         for li in range(4)]

resB = np.load('results_methodB.npy', allow_pickle=True).item()
r2_B  = np.array(resB['r2'])
rms_B = np.array(resB['rmse'])

resC_raw = np.load('results_methodC.npy', allow_pickle=True).item()
r2_C  = np.max([[v['r2'][li]   for li in range(4)] for v in resC_raw.values()], axis=0)
rms_C = [resC_raw[max(resC_raw, key=lambda k: resC_raw[k]['r2'][li])]['rmse'][li]
         for li in range(4)]

resD = np.load('results_methodD.npy', allow_pickle=True).item()
r2_D  = np.array(resD['r2'])
rms_D = np.array(resD['rmse'])

methods = {
    'A: Handcrafted\n+ ML':   {'r2': r2_A,  'rmse': rms_A,  'color': 'steelblue'},
    'B: 1D CNN':               {'r2': r2_B,  'rmse': rms_B,  'color': 'darkorange'},
    'C: FFT\n+ ML':            {'r2': r2_C,  'rmse': rms_C,  'color': 'seagreen'},
    'D: STFT\n+ 2D CNN':       {'r2': r2_D,  'rmse': rms_D,  'color': 'purple'},
}

# ── Print table ────────────────────────────────────────────────────────────
print(f'\n{"Method":<22}  {"S R²":>6} {"S RMSE":>7}  {"T R²":>6} {"T RMSE":>7}  '
      f'{"M R²":>6} {"M RMSE":>7}  {"B R²":>6} {"B RMSE":>7}  {"Avg R²":>7}')
print('-' * 100)
for mname, res in methods.items():
    line = f'{mname.replace(chr(10)," "):<22}  '
    for li in range(4):
        line += f'{res["r2"][li]:>6.3f} {res["rmse"][li]:>7.2f}%  '
    line += f'{np.mean(res["r2"]):>7.3f}'
    print(line)

# ── Plot 1: R² grouped by layer ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
x = np.arange(4); width = 0.18
for mi, (mname, res) in enumerate(methods.items()):
    ax.bar(x + mi * width, res['r2'], width,
           label=mname.replace('\n', ' '), color=res['color'], alpha=0.85)
ax.set_xticks(x + 1.5 * width); ax.set_xticklabels(LAYERS, fontsize=12)
ax.set_ylabel('R²', fontsize=12); ax.set_ylim([0, 1.05])
ax.set_title('R² by moisture depth layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
ax.axhline(0.8, color='red', lw=1, ls='--', alpha=0.5, label='R²=0.8')

ax = axes[1]
for mi, (mname, res) in enumerate(methods.items()):
    ax.bar(x + mi * width, res['rmse'], width,
           label=mname.replace('\n', ' '), color=res['color'], alpha=0.85)
ax.set_xticks(x + 1.5 * width); ax.set_xticklabels(LAYERS, fontsize=12)
ax.set_ylabel('RMSE (%)', fontsize=12)
ax.set_title('RMSE by moisture depth layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('GPR → Soil Moisture Prediction: Method Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_summary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Radar / average R² bar ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
avg_r2  = [np.mean(res['r2'])   for res in methods.values()]
avg_rms = [np.mean(res['rmse']) for res in methods.values()]
colors  = [res['color'] for res in methods.values()]
labels  = [m.replace('\n', ' ') for m in methods.keys()]
bars = ax.bar(labels, avg_r2, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
for bar, r2 in zip(bars, avg_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{r2:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Average R² (across S/T/M/B)', fontsize=12)
ax.set_ylim([0, 1.1])
ax.set_title('Average R² — all methods', fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_summary_avg_r2.png', dpi=150, bbox_inches='tight')
plt.close()

print('\nSaved: fig_summary_comparison.png, fig_summary_avg_r2.png')
