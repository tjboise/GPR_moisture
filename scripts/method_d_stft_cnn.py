"""
Method D: STFT (tuned parameters) + 2D CNN → moisture regression
Evaluation: 5-fold cross-validation (stratified by condition)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

LAYERS = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm = np.load('../data/X_norm.npy')       # (135, 256)
y      = np.load('../data/y_moisture.npy')   # (135, 4)
cond   = np.load('../data/cond_labels.npy')  # (135,) — used for stratification

# ── STFT parameters ────────────────────────────────────────────────────────
FS_HZ    = 1e9 / 0.099609
NPERSEG  = 32
NOVERLAP = 28
NFFT     = 512

def make_stft_image(trace):
    f, t, Zxx = stft(trace, fs=FS_HZ, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)
    f_mask = f <= 3e9
    img = mag[f_mask, :]
    img = img / (img.max() + 1e-12)
    return img.astype(np.float32)

# Pre-compute all STFT images
sample_img = make_stft_image(X_norm[0])
H, W = sample_img.shape
print(f'STFT image shape: {H} × {W}  (freq × time)')

imgs = np.stack([make_stft_image(X_norm[i]) for i in range(len(X_norm))])
imgs = imgs[:, None, :, :]  # (N, 1, H, W)


# ── Model definition ───────────────────────────────────────────────────────
class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


def train_fold(X_tr, y_tr, X_te, y_te, fold_idx):
    y_mean = torch.tensor(y_tr.mean(0), dtype=torch.float32)
    y_std  = torch.tensor(y_tr.std(0),  dtype=torch.float32).clamp(min=1e-6)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_tr_n = (torch.tensor(y_tr, dtype=torch.float32) - y_mean) / y_std
    y_te_n = (torch.tensor(y_te, dtype=torch.float32) - y_mean) / y_std

    dl = DataLoader(TensorDataset(X_tr_t, y_tr_n), batch_size=16, shuffle=True)

    model = CNN2D()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    patience = 60
    train_losses, val_losses = [], []

    for epoch in range(600):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr_t)

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t), y_te_n).item()

        train_losses.append(ep_loss); val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
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

    y_pred = y_pred_n * y_std.numpy() + y_mean.numpy()
    return y_pred, train_losses, val_losses


# ── 5-fold cross-validation ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

all_preds  = np.full_like(y, np.nan)   # collect OOF predictions
fold_r2s   = []                        # (5, 4) — R² per fold per layer
fold_rmses = []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(imgs, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
    y_pred, tl, vl = train_fold(
        imgs[tr_idx], y[tr_idx],
        imgs[te_idx], y[te_idx],
        fold_i
    )
    all_preds[te_idx] = y_pred

    r2s, rmses = [], []
    for li, lbl in enumerate(LAYERS):
        r2   = r2_score(y[te_idx, li], y_pred[:, li])
        rmse = np.sqrt(mean_squared_error(y[te_idx, li], y_pred[:, li]))
        r2s.append(r2); rmses.append(rmse)
    fold_r2s.append(r2s); fold_rmses.append(rmses)
    print(f'  R²: S={r2s[0]:.3f} T={r2s[1]:.3f} M={r2s[2]:.3f} B={r2s[3]:.3f}  avg={np.mean(r2s):.3f}')

fold_r2s   = np.array(fold_r2s)    # (5, 4)
fold_rmses = np.array(fold_rmses)  # (5, 4)

print(f'\n{"Layer":<5} {"Mean R²":>8} {"Std R²":>7}  {"Mean RMSE":>10} {"Std RMSE":>9}')
print('-' * 46)
r2_mean, r2_std = fold_r2s.mean(0), fold_r2s.std(0)
rm_mean, rm_std = fold_rmses.mean(0), fold_rmses.std(0)
for li, lbl in enumerate(LAYERS):
    print(f'{lbl:<5} {r2_mean[li]:>8.3f} {r2_std[li]:>7.3f}  {rm_mean[li]:>9.2f}% {rm_std[li]:>8.2f}%')
print(f'\nAvg R²: {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')

# ── OOF scatter plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Method D — STFT + 2D CNN: OOF Predictions (5-fold CV, n=135)',
             fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y[:, li], all_preds[:, li], color='purple',
               edgecolors='white', s=40, alpha=0.75)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl} layer\nR²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual moisture (%)'); ax.set_ylabel('Predicted moisture (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig_methodD_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── STFT sample images — one low + one high per layer ─────────────────────
LAYER_LABELS = ['S (0 cm)', 'T (8 cm)', 'M (22 cm)', 'B (35 cm)']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle(f'STFT images (nperseg={NPERSEG}, noverlap={NOVERLAP}, nfft={NFFT})\n'
             'Top row: low moisture example  |  Bottom row: high moisture example',
             fontsize=11, fontweight='bold')

for col, (li, lname) in enumerate(zip(range(4), LAYER_LABELS)):
    layer_mo  = y[:, li]
    sorted_i  = np.argsort(layer_mo)
    idx_low   = sorted_i[0]       # lowest moisture for this layer
    idx_high  = sorted_i[-1]      # highest moisture for this layer

    for row, (idx, tag) in enumerate([(idx_low, 'low'), (idx_high, 'high')]):
        ax = axes[row, col]
        ax.imshow(imgs[idx, 0], aspect='auto', origin='lower', cmap='hot',
                  extent=[0, W, 0, H])
        ax.set_title(f'{lname}\n{layer_mo[idx]:.1f}% ({tag})', fontsize=9)
        ax.set_xlabel('Time bins', fontsize=8)
        ax.set_ylabel('Freq bins', fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('../results/fig_methodD_stft_samples.png', dpi=150, bbox_inches='tight')
plt.close()

np.save('../results/results_methodD.npy',
        {'r2': r2_mean.tolist(), 'rmse': rm_mean.tolist(),
         'r2_std': r2_std.tolist(), 'rmse_std': rm_std.tolist(),
         'fold_r2s': fold_r2s.tolist(), 'fold_rmses': fold_rmses.tolist()},
        allow_pickle=True)
print('\nSaved: fig_methodD_scatter.png, fig_methodD_stft_samples.png, results_methodD.npy')
