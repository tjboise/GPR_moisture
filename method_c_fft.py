"""
Method C: FFT magnitude spectrum as features + ML regression
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

LAYERS = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42

X_norm  = np.load('X_norm.npy')       # (112, 256)
y       = np.load('y_moisture.npy')   # (112, 4)
idx_tr  = np.load('idx_train.npy')
idx_te  = np.load('idx_test.npy')

# ── FFT feature extraction ─────────────────────────────────────────────────
def fft_features(X):
    """FFT magnitude spectrum (one-sided), shape (N, 129)."""
    F = np.abs(np.fft.rfft(X, axis=1))          # (N, 129)
    F = F / (F.max(axis=1, keepdims=True) + 1e-12)   # normalize per sample
    return F.astype(np.float32)

F = fft_features(X_norm)
print(f'FFT feature shape: {F.shape}')

F_tr, F_te = F[idx_tr], F[idx_te]
y_tr, y_te = y[idx_tr], y[idx_te]

scaler = StandardScaler()
F_tr_s = scaler.fit_transform(F_tr)
F_te_s = scaler.transform(F_te)

models = {
    'Random Forest': MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)),
    'SVR (RBF)': MultiOutputRegressor(
        SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.5)),
    'Gradient Boosting': MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                  random_state=RANDOM_SEED)),
}

results = {}
print(f'\n{"Model":<22} {"Layer":<5} {"R²":>6} {"RMSE":>7}')
print('-' * 44)
for mname, model in models.items():
    model.fit(F_tr_s, y_tr)
    y_pred = model.predict(F_te_s)
    r2s, rmses = [], []
    for li, lbl in enumerate(LAYERS):
        r2   = r2_score(y_te[:, li], y_pred[:, li])
        rmse = np.sqrt(mean_squared_error(y_te[:, li], y_pred[:, li]))
        r2s.append(r2); rmses.append(rmse)
        print(f'{mname:<22} {lbl:<5} {r2:>6.3f} {rmse:>7.2f}%')
    results[mname] = {'r2': r2s, 'rmse': rmses, 'y_pred': y_pred}
    print()

# ── Scatter plots for best model ──────────────────────────────────────────
best = max(results, key=lambda k: np.mean(results[k]['r2']))
y_pred_best = results[best]['y_pred']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f'Method C — FFT + {best}: Predicted vs Actual', fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y_te[:, li], y_pred_best[:, li], color='seagreen', edgecolors='white', s=60)
    mn = min(y_te[:, li].min(), y_pred_best[:, li].min()) - 1
    mx = max(y_te[:, li].max(), y_pred_best[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    r2   = results[best]['r2'][li]
    rmse = results[best]['rmse'][li]
    ax.set_title(f'{lbl} layer\nR²={r2:.3f}, RMSE={rmse:.2f}%', fontsize=10)
    ax.set_xlabel('Actual moisture (%)'); ax.set_ylabel('Predicted moisture (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodC_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── R² comparison ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4); width = 0.25
colors = ['steelblue', 'darkorange', 'seagreen']
for mi, (mname, res) in enumerate(results.items()):
    ax.bar(x + mi * width, res['r2'], width, label=mname, color=colors[mi], alpha=0.85)
ax.set_xticks(x + width); ax.set_xticklabels(LAYERS)
ax.set_ylabel('R²'); ax.set_ylim([0, 1])
ax.set_title('Method C (FFT features): R² by layer and model', fontweight='bold')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodC_r2.png', dpi=150, bbox_inches='tight')
plt.close()

np.save('results_methodC.npy', {
    k: {'r2': v['r2'], 'rmse': v['rmse']} for k, v in results.items()
}, allow_pickle=True)
print('Saved: fig_methodC_scatter.png, fig_methodC_r2.png')
