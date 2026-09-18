"""
Method D — Augmented: STFT + 2D CNN with Gaussian noise + Mixup
Evaluation: 5-fold cross-validation (stratified by condition)
Baseline (no augmentation): Avg R² = 0.797 ± 0.056
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
from torch.utils.data import Dataset, DataLoader

LAYERS = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm = np.load('../data/X_norm.npy')      # (135, 256)  raw A-scans
y      = np.load('../data/y_moisture.npy')  # (135, 4)
cond   = np.load('../data/cond_labels.npy')

# ── STFT parameters (unchanged) ────────────────────────────────────────────
FS_HZ    = 1e9 / 0.099609
NPERSEG  = 32
NOVERLAP = 28
NFFT     = 512

def ascan_to_stft(trace):
    f, _, Zxx = stft(trace, fs=FS_HZ, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= 3e9, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)

sample_img = ascan_to_stft(X_norm[0])
H, W = sample_img.shape
print(f'STFT image: {H} × {W}  |  Augmentation: Gaussian noise only (σ=0.02, p=0.5)')

# ── Augmentation parameters ────────────────────────────────────────────────
NOISE_SIGMA  = 0.02   # Gaussian noise std (signal is in [-1, 1])
NOISE_PROB   = 0.5    # probability of applying noise per sample per epoch
MIXUP_PROB   = 0.0    # 0.0 = noise only (no Mixup)


# ── Dataset: computes STFT on-the-fly so noise is applied to A-scan first ──
class GPRDataset(Dataset):
    def __init__(self, X, y_norm, augment=False):
        self.X       = X          # raw A-scans (N, 256)
        self.y       = y_norm     # normalized labels
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        trace = self.X[idx].copy()
        if self.augment and np.random.rand() < NOISE_PROB:
            trace += np.random.normal(0, NOISE_SIGMA, size=trace.shape)
            trace = np.clip(trace, -1.0, 1.0)
        img = ascan_to_stft(trace)
        return torch.tensor(img[None], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)


def mixup_batch(x, y, alpha=0.2):
    """Apply Mixup to a batch: mix randomly paired samples."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], \
           lam * y + (1 - lam) * y[idx]


# ── Model (same as baseline) ───────────────────────────────────────────────
class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 4),
        )
    def forward(self, x):
        return self.head(self.encoder(x))


def train_fold(X_tr, y_tr, X_te, y_te, fold_idx):
    y_mean = torch.tensor(y_tr.mean(0), dtype=torch.float32)
    y_std  = torch.tensor(y_tr.std(0),  dtype=torch.float32).clamp(min=1e-6)

    y_tr_n = ((torch.tensor(y_tr, dtype=torch.float32) - y_mean) / y_std).numpy()
    y_te_n = ((torch.tensor(y_te, dtype=torch.float32) - y_mean) / y_std).numpy()

    train_ds = GPRDataset(X_tr, y_tr_n, augment=True)
    val_ds   = GPRDataset(X_te, y_te_n, augment=False)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Build full validation tensors (no augmentation)
    X_te_t = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    y_te_t = torch.stack([val_ds[i][1] for i in range(len(val_ds))])

    model = CNN2D()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    patience = 60

    for epoch in range(600):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            if np.random.rand() < MIXUP_PROB:
                xb, yb = mixup_batch(xb, yb)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t), y_te_t).item()

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
    return y_pred


# ── 5-fold cross-validation ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

all_preds  = np.full_like(y, np.nan)
fold_r2s   = []
fold_rmses = []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_norm, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
    y_pred = train_fold(X_norm[tr_idx], y[tr_idx],
                        X_norm[te_idx], y[te_idx], fold_i)
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
print(f'Baseline: 0.797 ± 0.056')
print(f'Delta:   {r2_mean.mean()-0.797:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Method D Augmented (Gaussian noise only) — 5-fold CV, n=135',
             fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y[:, li], all_preds[:, li], color='darkorchid',
               edgecolors='white', s=40, alpha=0.8)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl}  R²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig_methodD_aug_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

np.save('../results/results_methodD_aug.npy',
        {'r2': r2_mean.tolist(), 'rmse': rm_mean.tolist(),
         'r2_std': r2_std.tolist(), 'fold_r2s': fold_r2s.tolist()},
        allow_pickle=True)
print('\nSaved: fig_methodD_aug_scatter.png, results_methodD_aug.npy')
