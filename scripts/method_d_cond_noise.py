"""
Condition-aware (soil + pipe factored) + Gaussian noise augmentation
Saves per-fold synthetic A-scans to: data/augmented/noise/fold_k/
  X_synth.npy   — noisy A-scans, shape (n_synth, 256)
  y_synth.npy   — moisture labels, shape (n_synth, 4)
  soil_synth.npy / pipe_synth.npy — condition labels
  meta.csv      — human-readable summary

Baseline (no condition):    Avg R² = 0.797 ± 0.056
Cond-aware v2 (no aug):     Avg R² = 0.816 ± 0.056
"""
import numpy as np
import pandas as pd
import pathlib, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LAYERS      = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT    = pathlib.Path('../data/augmented/noise')
ROOT.mkdir(parents=True, exist_ok=True)

X_norm = np.load('../data/X_norm.npy')
y      = np.load('../data/y_moisture.npy')
cond   = np.load('../data/cond_labels.npy')

SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP = {0: 0, 1: 1, 2: 1}
soil_label = np.array([SOIL_MAP[c] for c in cond])
pipe_label = np.array([PIPE_MAP[c] for c in cond])
COND_NAMES = {0: 'Sand 2"', 1: 'Sand 4"', 2: 'Clay 4"'}
SOIL_NAMES = {0: 'Sand', 1: 'Clay'}
PIPE_NAMES = {0: '2"', 1: '4"'}

# ── STFT ──────────────────────────────────────────────────────────────────
FS_HZ    = 1e9 / 0.099609
NPERSEG  = 32
NOVERLAP = 28
NFFT     = 512

def ascan_to_stft(trace):
    f, _, Zxx = stft(trace, fs=FS_HZ, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= 3e9, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)

# ── Augmentation params ────────────────────────────────────────────────────
NOISE_SIGMA  = 0.02
N_SYNTH_RATIO = 2   # generate 2× training set size

# ── Dataset (raw A-scan + online STFT) ────────────────────────────────────
class GPRDataset(Dataset):
    def __init__(self, X_ascan, soil, pipe, y_norm, augment=False):
        self.X    = X_ascan
        self.soil = torch.tensor(soil, dtype=torch.long)
        self.pipe = torch.tensor(pipe, dtype=torch.long)
        self.y    = torch.tensor(y_norm, dtype=torch.float32)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        trace = self.X[i].copy()
        if self.augment:
            trace += np.random.normal(0, NOISE_SIGMA, size=trace.shape)
            trace  = np.clip(trace, -1.0, 1.0)
        img = ascan_to_stft(trace)
        return torch.tensor(img[None], dtype=torch.float32), self.soil[i], self.pipe[i], self.y[i]


# ── Model (same as cond v2) ────────────────────────────────────────────────
N_COND = 3
class CNN2DCondV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 16 + 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )
    def forward(self, x, soil, pipe):
        feat    = self.encoder(x).flatten(1)
        c_vec   = torch.cat([F.one_hot(soil, 2).float(),
                              F.one_hot(pipe, 2).float()], dim=1)
        return self.head(torch.cat([feat, c_vec], dim=1))


def generate_and_save_noise(X_tr, y_tr, soil_tr, pipe_tr, fold_idx):
    """Generate noisy A-scans, save to disk, return combined arrays."""
    n_real  = len(X_tr)
    n_synth = n_real * N_SYNTH_RATIO
    rng     = np.random.default_rng(RANDOM_SEED + fold_idx)

    # Draw with replacement from training set
    src_idx  = rng.integers(0, n_real, n_synth)
    X_synth  = X_tr[src_idx].copy()
    noise    = rng.normal(0, NOISE_SIGMA, X_synth.shape)
    X_synth  = np.clip(X_synth + noise, -1.0, 1.0).astype(np.float32)
    y_synth  = y_tr[src_idx].copy()
    soil_syn = soil_tr[src_idx].copy()
    pipe_syn = pipe_tr[src_idx].copy()

    # Save to disk
    fold_dir = ROOT / f'fold_{fold_idx+1}'
    fold_dir.mkdir(exist_ok=True)
    np.save(fold_dir / 'X_synth.npy',    X_synth)
    np.save(fold_dir / 'y_synth.npy',    y_synth)
    np.save(fold_dir / 'soil_synth.npy', soil_syn)
    np.save(fold_dir / 'pipe_synth.npy', pipe_syn)

    # Human-readable CSV
    df = pd.DataFrame({
        'src_idx':   src_idx,
        'soil':      [SOIL_NAMES[s] for s in soil_syn],
        'pipe':      [PIPE_NAMES[p] for p in pipe_syn],
        'S_pct':     y_synth[:, 0].round(2),
        'T_pct':     y_synth[:, 1].round(2),
        'M_pct':     y_synth[:, 2].round(2),
        'B_pct':     y_synth[:, 3].round(2),
        'noise_sigma': NOISE_SIGMA,
    })
    df.to_csv(fold_dir / 'meta.csv', index_label='synth_idx')
    print(f'  Saved {n_synth} synthetic samples → {fold_dir}')

    return X_synth, y_synth, soil_syn, pipe_syn


def train_fold(X_tr, soil_tr, pipe_tr, y_tr,
               X_te, soil_te, pipe_te, y_te, fold_idx):
    # Generate & save noise-augmented A-scans
    X_syn, y_syn, soil_syn, pipe_syn = generate_and_save_noise(
        X_tr, y_tr, soil_tr, pipe_tr, fold_idx)

    # Combine real + synthetic
    X_aug    = np.concatenate([X_tr, X_syn])
    y_aug    = np.concatenate([y_tr, y_syn])
    soil_aug = np.concatenate([soil_tr, soil_syn])
    pipe_aug = np.concatenate([pipe_tr, pipe_syn])

    y_mean = torch.tensor(y_aug.mean(0), dtype=torch.float32)
    y_std  = torch.tensor(y_aug.std(0),  dtype=torch.float32).clamp(min=1e-6)

    y_aug_n = ((torch.tensor(y_aug, dtype=torch.float32) - y_mean) / y_std).numpy()
    y_te_n  = ((torch.tensor(y_te,  dtype=torch.float32) - y_mean) / y_std).numpy()

    # NOTE: augmented dataset uses pre-generated noisy X, no further online aug
    tr_ds = GPRDataset(X_aug, soil_aug, pipe_aug, y_aug_n, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True)

    te_ds = GPRDataset(X_te, soil_te, pipe_te, y_te_n, augment=False)
    X_te_t    = torch.stack([te_ds[i][0] for i in range(len(te_ds))])
    soil_te_t = torch.tensor(soil_te, dtype=torch.long)
    pipe_te_t = torch.tensor(pipe_te, dtype=torch.long)
    y_te_t    = torch.tensor(y_te_n,  dtype=torch.float32)

    model = CNN2DCondV2()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30, min_lr=1e-5)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    patience = 80
    for epoch in range(800):
        model.train()
        for xb, sb, pb, yb in tr_dl:
            opt.zero_grad(); crit(model(xb, sb, pb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t, soil_te_t, pipe_te_t), y_te_t).item()
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'  CNN early stop epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_n = model(X_te_t, soil_te_t, pipe_te_t).numpy()

    return y_pred_n * y_std.numpy() + y_mean.numpy()


# ── 5-fold CV ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
all_preds  = np.full_like(y, np.nan)
fold_r2s, fold_rmses = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_norm, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)} real + {len(tr_idx)*N_SYNTH_RATIO} synth, test={len(te_idx)})')
    y_pred = train_fold(
        X_norm[tr_idx], soil_label[tr_idx], pipe_label[tr_idx], y[tr_idx],
        X_norm[te_idx], soil_label[te_idx], pipe_label[te_idx], y[te_idx],
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

fold_r2s   = np.array(fold_r2s)
fold_rmses = np.array(fold_rmses)
r2_mean, r2_std = fold_r2s.mean(0), fold_r2s.std(0)
rm_mean, rm_std = fold_rmses.mean(0), fold_rmses.std(0)

print(f'\n{"Layer":<5} {"Mean R²":>8} {"Std":>6}  {"RMSE":>8} {"Std":>6}')
print('-' * 42)
for li, lbl in enumerate(LAYERS):
    print(f'{lbl:<5} {r2_mean[li]:>8.3f} {r2_std[li]:>6.3f}  {rm_mean[li]:>7.2f}% {rm_std[li]:>5.2f}%')
print(f'\nAvg R²: {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')
print(f'Baseline (no cond):         0.797 ± 0.056')
print(f'Cond-aware v2 (no aug):     0.816 ± 0.056')
print(f'Cond-aware v2 + noise:     {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')
print(f'Delta vs cond v2: {r2_mean.mean()-0.816:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
COND_COLORS = {0: '#e07b39', 1: '#4a90d9', 2: '#5cb85c'}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Condition-aware (soil+pipe) + Noise aug — 5-fold CV, n=135',
             fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    for ci, cname in COND_NAMES.items():
        mask = cond == ci
        ax.scatter(y[mask, li], all_preds[mask, li],
                   color=COND_COLORS[ci], edgecolors='white', s=45, alpha=0.85, label=cname)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl}  R²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
    if li == 0: ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('../results/fig_cond_noise_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: fig_cond_noise_scatter.png')
