"""
Option A: Hand-crafted physical features + Random Forest / SVR
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

LAYERS = ['S', 'T', 'M', 'B']

X_norm   = np.load('X_norm.npy')          # (112, 256), normalized traces
y        = np.load('y_moisture.npy')      # (112, 4)
idx_tr   = np.load('idx_train.npy')
idx_te   = np.load('idx_test.npy')
time_ns  = np.load('time_ns.npy')
dt       = time_ns[1] - time_ns[0]

# ── Feature extraction ────────────────────────────────────────────────────
def extract_features(traces):
    """Extract physical features from each normalized A-scan."""
    feats = []
    for i in range(len(traces)):
        tr = traces[i]
        env = np.abs(hilbert(tr))               # envelope

        # 1. Peak amplitude (max absolute)
        peak_amp = np.max(np.abs(tr))

        # 2. Peak-to-peak
        pk2pk = tr.max() - tr.min()

        # 3. Signal energy (sum of squares)
        energy = np.sum(tr ** 2)

        # 4. Envelope maximum
        env_max = env.max()

        # 5. Envelope area (integral)
        env_area = np.trapz(env) * dt

        # 6. Time of envelope peak
        t_env_peak = time_ns[np.argmax(env)]

        # 7. Dominant frequency (FFT)
        fs = 1.0 / (dt * 1e-9)              # Hz
        fft_mag = np.abs(np.fft.rfft(tr))
        freqs   = np.fft.rfftfreq(len(tr), d=dt * 1e-9)
        dom_freq = freqs[np.argmax(fft_mag)] / 1e9  # GHz

        # 8. Spectral centroid
        spec_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12) / 1e9

        # 9. Signal energy in first half vs second half of time window
        mid = len(tr) // 2
        ratio_first_half = np.sum(tr[:mid] ** 2) / (energy + 1e-12)

        # 10. RMS amplitude
        rms = np.sqrt(np.mean(tr ** 2))

        # 11. Zero-crossing rate
        zc = np.sum(np.diff(np.sign(tr)) != 0)

        # 12–14. Envelope at 3 fixed depth windows (early / mid / late)
        n = len(tr)
        env_early = env[:n//3].mean()
        env_mid   = env[n//3:2*n//3].mean()
        env_late  = env[2*n//3:].mean()

        feats.append([
            peak_amp, pk2pk, energy, env_max, env_area,
            t_env_peak, dom_freq, spec_centroid,
            ratio_first_half, rms, zc,
            env_early, env_mid, env_late,
        ])
    return np.array(feats, dtype=np.float32)


print('Extracting features...')
F = extract_features(X_norm)
print(f'Feature matrix shape: {F.shape}')

F_tr, F_te = F[idx_tr], F[idx_te]
y_tr, y_te = y[idx_tr], y[idx_te]

scaler = StandardScaler()
F_tr_s = scaler.fit_transform(F_tr)
F_te_s = scaler.transform(F_te)

# ── Models ────────────────────────────────────────────────────────────────
models = {
    'Random Forest': MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=42)),
    'SVR (RBF)': MultiOutputRegressor(
        SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.5)),
    'Gradient Boosting': MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)),
}

results = {}
print(f'\n{"Model":<22} {"Layer":<5} {"R²":>6} {"RMSE":>7}')
print('-' * 44)

for model_name, model in models.items():
    model.fit(F_tr_s, y_tr)
    y_pred = model.predict(F_te_s)
    r2s, rmses = [], []
    for li, lbl in enumerate(LAYERS):
        r2   = r2_score(y_te[:, li], y_pred[:, li])
        rmse = np.sqrt(mean_squared_error(y_te[:, li], y_pred[:, li]))
        r2s.append(r2); rmses.append(rmse)
        print(f'{model_name:<22} {lbl:<5} {r2:>6.3f} {rmse:>7.2f}%')
    results[model_name] = {'r2': r2s, 'rmse': rmses, 'y_pred': y_pred}
    print()

# ── Feature importance (Random Forest) ───────────────────────────────────
feat_names = [
    'Peak amp', 'Peak-to-peak', 'Energy', 'Env max', 'Env area',
    'T env peak', 'Dom freq', 'Spec centroid',
    'Ratio 1st half', 'RMS', 'Zero-cross rate',
    'Env early', 'Env mid', 'Env late',
]
rf = models['Random Forest'].estimators_[0]   # estimator for S layer
importances = np.mean(
    [models['Random Forest'].estimators_[li].feature_importances_
     for li in range(4)], axis=0)

# ── Plot 1: Scatter plots (predicted vs actual) for best model ────────────
best_model = 'Random Forest'
y_pred_best = results[best_model]['y_pred']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f'Option A — {best_model}: Predicted vs Actual moisture', fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y_te[:, li], y_pred_best[:, li], color='steelblue', edgecolors='white', s=60)
    mn = min(y_te[:, li].min(), y_pred_best[:, li].min()) - 1
    mx = max(y_te[:, li].max(), y_pred_best[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    r2   = results[best_model]['r2'][li]
    rmse = results[best_model]['rmse'][li]
    ax.set_title(f'{lbl} layer\nR²={r2:.3f}, RMSE={rmse:.2f}%', fontsize=10)
    ax.set_xlabel('Actual moisture (%)')
    ax.set_ylabel('Predicted moisture (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodA_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Feature importance ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
order = np.argsort(importances)[::-1]
ax.bar(range(len(feat_names)), importances[order], color='steelblue')
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels([feat_names[o] for o in order], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean feature importance (avg over S/T/M/B)')
ax.set_title('Option A — Random Forest: Feature importance', fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodA_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: R² comparison across models and layers ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4)
width = 0.25
colors = ['steelblue', 'darkorange', 'seagreen']
for mi, (mname, res) in enumerate(results.items()):
    ax.bar(x + mi * width, res['r2'], width, label=mname, color=colors[mi], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(LAYERS)
ax.set_ylabel('R²')
ax.set_ylim([0, 1])
ax.set_title('Option A: R² by moisture layer and model', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodA_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Save results for final comparison
np.save('results_methodA.npy', {
    k: {'r2': v['r2'], 'rmse': v['rmse']} for k, v in results.items()
}, allow_pickle=True)

print('Saved: fig_methodA_scatter.png, fig_methodA_importance.png, fig_methodA_r2_comparison.png')
