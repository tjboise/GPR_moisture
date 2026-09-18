"""
1D CNN: raw A-scan (256 samples) → moisture regression at 4 depths
Evaluation: 5-fold stratified CV (same split as baseline)
Baseline (STFT + 2D CNN): Avg R² = 0.797 ± 0.056
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

LAYERS      = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm = np.load('../data/X_norm.npy')       # (135, 256)
y      = np.load('../data/y_moisture.npy')   # (135, 4)
cond   = np.load('../data/cond_labels.npy')  # (135,)

print(f'Dataset: {X_norm.shape[0]} samples, {X_norm.shape[1]} time samples per A-scan')


# ── Model ──────────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),   # → (16, 128)
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # → (32, 64)
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),  # → (64, 32)
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                                                   # → (128, 8)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )
    def forward(self, x):
        return self.head(self.encoder(x))


def train_fold(X_tr, y_tr, X_te, y_te, fold_idx):
    y_mean = torch.tensor(y_tr.mean(0), dtype=torch.float32)
    y_std  = torch.tensor(y_tr.std(0),  dtype=torch.float32).clamp(min=1e-6)

    # shape: (N, 1, 256)
    X_tr_t = torch.tensor(X_tr[:, None, :], dtype=torch.float32)
    X_te_t = torch.tensor(X_te[:, None, :], dtype=torch.float32)
    y_tr_n = (torch.tensor(y_tr, dtype=torch.float32) - y_mean) / y_std
    y_te_n = (torch.tensor(y_te, dtype=torch.float32) - y_mean) / y_std

    dl = DataLoader(TensorDataset(X_tr_t, y_tr_n), batch_size=16, shuffle=True)

    model = CNN1D()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30, min_lr=1e-5)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    patience = 80

    for epoch in range(800):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t), y_te_n).item()

        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'  Fold {fold_idx+1}: early stop epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_n = model(X_te_t).numpy()

    return y_pred_n * y_std.numpy() + y_mean.numpy()


# ── 5-fold CV ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
all_preds  = np.full_like(y, np.nan)
fold_r2s, fold_rmses = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_norm, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
    y_pred = train_fold(X_norm[tr_idx], y[tr_idx],
                        X_norm[te_idx],  y[te_idx], fold_i)
    all_preds[te_idx] = y_pred

    r2s, rmses = [], []
    for li, lbl in enumerate(LAYERS):
        r2   = r2_score(y[te_idx, li], y_pred[:, li])
        rmse = np.sqrt(mean_squared_error(y[te_idx, li], y_pred[:, li]))
        r2s.append(r2); rmses.append(rmse)
    fold_r2s.append(r2s); fold_rmses.append(rmses)
    print(f'  R²: S={r2s[0]:.3f} T={r2s[1]:.3f} M={r2s[2]:.3f} B={r2s[3]:.3f}  avg={np.mean(r2s):.3f}')

fold_r2s   = np.array(fold_r2s)
fold_rmses = np.array(fold_rmses)
r2_mean, r2_std = fold_r2s.mean(0), fold_r2s.std(0)
rm_mean, rm_std = fold_rmses.mean(0), fold_rmses.std(0)

print(f'\n{"Layer":<5} {"Mean R²":>8} {"Std":>6}  {"RMSE":>8} {"Std":>6}')
print('-' * 42)
for li, lbl in enumerate(LAYERS):
    print(f'{lbl:<5} {r2_mean[li]:>8.3f} {r2_std[li]:>6.3f}  {rm_mean[li]:>7.2f}% {rm_std[li]:>5.2f}%')
print(f'\nAvg R²: {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')
print(f'Baseline (STFT+2D CNN): 0.797 ± 0.056')
print(f'Delta:  {r2_mean.mean()-0.797:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('1D CNN (raw A-scan) — 5-fold CV, n=135', fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y[:, li], all_preds[:, li], color='seagreen',
               edgecolors='white', s=40, alpha=0.8)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl}  R²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig_1dcnn_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: fig_1dcnn_scatter.png')
