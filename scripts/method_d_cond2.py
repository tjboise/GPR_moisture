"""
Method D — Condition-aware v2: STFT + 2D CNN
Condition encoding: soil one-hot(2) + pipe-diameter one-hot(2) = 4 dims (factored)
vs. v1: single 3-class one-hot = 3 dims (joint)

Factored encoding gives the model separate "knobs" for soil type and pipe size,
so it can learn their effects independently rather than treating all 3 conditions
as unrelated categories.

Mapping:
  cond 0 → Sand 2"  : soil=Sand(0), pipe=2"(0)
  cond 1 → Sand 4"  : soil=Sand(0), pipe=4"(1)
  cond 2 → Clay 4"  : soil=Clay(1), pipe=4"(1)

Baseline (no condition):  Avg R² = 0.797 ± 0.056
v1 (joint 3-class):       Avg R² = 0.804 ± 0.061
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LAYERS      = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm = np.load('../data/X_norm.npy')
y      = np.load('../data/y_moisture.npy')
cond   = np.load('../data/cond_labels.npy')  # {0,1,2}

# ── Derived binary labels ──────────────────────────────────────────────────
# soil: 0=Sand, 1=Clay
SOIL_MAP = {0: 0, 1: 0, 2: 1}
# pipe diameter: 0=2", 1=4"
PIPE_MAP = {0: 0, 1: 1, 2: 1}

soil_label = np.array([SOIL_MAP[c] for c in cond])  # (135,)
pipe_label = np.array([PIPE_MAP[c] for c in cond])  # (135,)

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

imgs = np.stack([ascan_to_stft(X_norm[i]) for i in range(len(X_norm))])
H, W = imgs.shape[1], imgs.shape[2]
COND_DIM = 4  # soil one-hot(2) + pipe one-hot(2)
print(f'STFT image: {H}×{W}  |  Factored condition: soil_OH(2) + pipe_OH(2) = {COND_DIM} dims')


# ── Dataset ────────────────────────────────────────────────────────────────
class GPRDataset(Dataset):
    def __init__(self, imgs, soil, pipe, y_norm):
        self.imgs = torch.tensor(imgs[:, None, :, :], dtype=torch.float32)
        self.soil = torch.tensor(soil, dtype=torch.long)
        self.pipe = torch.tensor(pipe, dtype=torch.long)
        self.y    = torch.tensor(y_norm, dtype=torch.float32)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.soil[i], self.pipe[i], self.y[i]


# ── Model ──────────────────────────────────────────────────────────────────
class CNN2DCondV2(nn.Module):
    def __init__(self, cond_dim=COND_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        feat_dim = 64 * 4 * 4 + cond_dim  # 1024 + 4 = 1028
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x, soil, pipe):
        feat    = self.encoder(x).flatten(1)                        # (N, 1024)
        soil_oh = F.one_hot(soil, num_classes=2).float()            # (N, 2)
        pipe_oh = F.one_hot(pipe, num_classes=2).float()            # (N, 2)
        c_vec   = torch.cat([soil_oh, pipe_oh], dim=1)              # (N, 4)
        return self.head(torch.cat([feat, c_vec], dim=1))


def train_fold(imgs_tr, soil_tr, pipe_tr, y_tr,
               imgs_te, soil_te, pipe_te, y_te, fold_idx):
    y_mean = torch.tensor(y_tr.mean(0), dtype=torch.float32)
    y_std  = torch.tensor(y_tr.std(0),  dtype=torch.float32).clamp(min=1e-6)

    y_tr_n = ((torch.tensor(y_tr, dtype=torch.float32) - y_mean) / y_std).numpy()
    y_te_n = ((torch.tensor(y_te, dtype=torch.float32) - y_mean) / y_std).numpy()

    tr_ds = GPRDataset(imgs_tr, soil_tr, pipe_tr, y_tr_n)
    te_ds = GPRDataset(imgs_te, soil_te, pipe_te, y_te_n)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True)

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
                print(f'  Fold {fold_idx+1}: early stop epoch {epoch+1}')
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

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(imgs, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
    y_pred = train_fold(
        imgs[tr_idx], soil_label[tr_idx], pipe_label[tr_idx], y[tr_idx],
        imgs[te_idx], soil_label[te_idx], pipe_label[te_idx], y[te_idx],
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
print(f'Baseline (no condition):    0.797 ± 0.056')
print(f'v1 (joint 3-class):         0.804 ± 0.061')
print(f'v2 (soil+pipe factored):   {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')
print(f'Delta vs baseline:  {r2_mean.mean()-0.797:+.3f}')
print(f'Delta vs v1:        {r2_mean.mean()-0.804:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
COND_NAMES  = {0: 'Sand 2"', 1: 'Sand 4"', 2: 'Clay 4"'}
COND_COLORS = {0: '#e07b39', 1: '#4a90d9', 2: '#5cb85c'}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('STFT + 2D CNN (factored: soil + pipe) — 5-fold CV, n=135',
             fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    for ci, cname in COND_NAMES.items():
        mask = cond == ci
        ax.scatter(y[mask, li], all_preds[mask, li],
                   color=COND_COLORS[ci], edgecolors='white',
                   s=45, alpha=0.85, label=cname)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl}  R²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
    if li == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('../results/fig_methodD_cond2_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: fig_methodD_cond2_scatter.png')
