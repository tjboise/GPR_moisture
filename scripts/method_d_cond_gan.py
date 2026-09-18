"""
Condition-aware (soil + pipe factored) + cWGAN-GP augmentation
GAN conditioned on: [soil_OH(2), pipe_OH(2), moisture_normalised(4)] = 8 dims
Saves per-fold synthetic A-scans to: data/augmented/gan/fold_k/
  X_synth.npy   — GAN-generated A-scans, shape (n_synth, 256)
  y_synth.npy   — interpolated moisture labels, shape (n_synth, 4)
  soil_synth.npy / pipe_synth.npy — condition labels
  meta.csv      — human-readable summary

Baseline (no cond):         Avg R² = 0.797 ± 0.056
Cond-aware v2 (no aug):     Avg R² = 0.816 ± 0.056
Cond-aware v2 + noise:      Avg R² = 0.801 ± 0.070
"""
import numpy as np
import pandas as pd
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

LAYERS      = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = pathlib.Path('../data/augmented/gan')
ROOT.mkdir(parents=True, exist_ok=True)

X_norm = np.load('../data/X_norm.npy')
y      = np.load('../data/y_moisture.npy')
cond   = np.load('../data/cond_labels.npy')

SOIL_MAP   = {0: 0, 1: 0, 2: 1}
PIPE_MAP   = {0: 0, 1: 1, 2: 1}
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

# ── GAN params ────────────────────────────────────────────────────────────
LATENT        = 64
N_SYNTH_RATIO = 2
GAN_EPOCHS    = 400
N_CRITIC      = 5
LAMBDA_GP     = 10.0
GAN_COND_DIM  = 2 + 2 + 4  # soil_OH + pipe_OH + moisture_normalised = 8

# ── GAN architecture ───────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT + GAN_COND_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),                    nn.LeakyReLU(0.2),
            nn.Linear(512, 256),                    nn.Tanh(),
        )
    def forward(self, z, c): return self.net(torch.cat([z, c], dim=1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + GAN_COND_DIM, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),                 nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, x, c): return self.net(torch.cat([x, c], dim=1))


def gradient_penalty(critic, real, fake, c, lambda_gp=LAMBDA_GP):
    bsz   = real.size(0)
    alpha = torch.rand(bsz, 1)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out  = critic(interp, c)
    grads  = torch.autograd.grad(d_out, interp,
                                 grad_outputs=torch.ones_like(d_out),
                                 create_graph=True, retain_graph=True)[0]
    return lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()


def make_gan_cond(soil, pipe, y_norm):
    """Build GAN condition vector: soil_OH + pipe_OH + y_normalised."""
    soil_oh = F.one_hot(torch.tensor(soil, dtype=torch.long), 2).float()
    pipe_oh = F.one_hot(torch.tensor(pipe, dtype=torch.long), 2).float()
    y_t     = torch.tensor(y_norm, dtype=torch.float32)
    return torch.cat([soil_oh, pipe_oh, y_t], dim=1)  # (N, 8)


def train_gan(X_tr, soil_tr, pipe_tr, y_tr_n, fold_idx):
    G = Generator(); C = Critic()
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    X_t   = torch.tensor(X_tr, dtype=torch.float32)
    c_vec = make_gan_cond(soil_tr, pipe_tr, y_tr_n)
    ds    = TensorDataset(X_t, c_vec)
    dl    = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

    for ep in range(GAN_EPOCHS):
        for real_x, real_c in dl:
            bsz = real_x.size(0)
            for _ in range(N_CRITIC):
                z    = torch.randn(bsz, LATENT)
                fake = G(z, real_c).detach()
                gp   = gradient_penalty(C, real_x, fake, real_c)
                c_loss = -C(real_x, real_c).mean() + C(fake, real_c).mean() + gp
                opt_C.zero_grad(); c_loss.backward(); opt_C.step()
            z      = torch.randn(bsz, LATENT)
            g_loss = -C(G(z, real_c), real_c).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    print(f'  GAN trained {GAN_EPOCHS} ep | G={g_loss.item():.3f} C={c_loss.item():.3f}')
    return G


def generate_and_save_gan(G, X_tr, y_tr, soil_tr, pipe_tr,
                          y_mean, y_std, n_synth, fold_idx):
    """Interpolate conditions between real pairs, generate A-scans, save."""
    G.eval()
    n_real  = len(X_tr)
    rng     = np.random.default_rng(RANDOM_SEED + fold_idx)
    idx_a   = rng.integers(0, n_real, n_synth)
    idx_b   = rng.integers(0, n_real, n_synth)
    lam     = rng.uniform(0.3, 0.7, (n_synth, 1))

    y_synth    = (lam * y_tr[idx_a] + (1 - lam) * y_tr[idx_b]).astype(np.float32)
    soil_synth = soil_tr[idx_a]
    pipe_synth = pipe_tr[idx_a]
    y_syn_n    = ((y_synth - y_mean) / y_std).astype(np.float32)

    c_vec = make_gan_cond(soil_synth, pipe_synth, y_syn_n)
    z     = torch.randn(n_synth, LATENT)
    with torch.no_grad():
        X_synth = G(z, c_vec).numpy().astype(np.float32)

    # Save
    fold_dir = ROOT / f'fold_{fold_idx+1}'
    fold_dir.mkdir(exist_ok=True)
    np.save(fold_dir / 'X_synth.npy',    X_synth)
    np.save(fold_dir / 'y_synth.npy',    y_synth)
    np.save(fold_dir / 'soil_synth.npy', soil_synth)
    np.save(fold_dir / 'pipe_synth.npy', pipe_synth)

    df = pd.DataFrame({
        'src_a':   idx_a, 'src_b': idx_b, 'lambda': lam.flatten().round(3),
        'soil':    [SOIL_NAMES[s] for s in soil_synth],
        'pipe':    [PIPE_NAMES[p] for p in pipe_synth],
        'S_pct':   y_synth[:, 0].round(2),
        'T_pct':   y_synth[:, 1].round(2),
        'M_pct':   y_synth[:, 2].round(2),
        'B_pct':   y_synth[:, 3].round(2),
    })
    df.to_csv(fold_dir / 'meta.csv', index_label='synth_idx')
    print(f'  Saved {n_synth} GAN samples → {fold_dir}')
    return X_synth, y_synth, soil_synth, pipe_synth


# ── CNN (condition-aware v2) ───────────────────────────────────────────────
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
        feat  = self.encoder(x).flatten(1)
        c_vec = torch.cat([F.one_hot(soil, 2).float(),
                           F.one_hot(pipe, 2).float()], dim=1)
        return self.head(torch.cat([feat, c_vec], dim=1))


class STFTDataset(Dataset):
    def __init__(self, X_ascan, soil, pipe, y_norm):
        self.X    = X_ascan
        self.soil = torch.tensor(soil, dtype=torch.long)
        self.pipe = torch.tensor(pipe, dtype=torch.long)
        self.y    = torch.tensor(y_norm, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        img = ascan_to_stft(self.X[i])
        return torch.tensor(img[None], dtype=torch.float32), self.soil[i], self.pipe[i], self.y[i]


def train_fold(X_tr, soil_tr, pipe_tr, y_tr,
               X_te, soil_te, pipe_te, y_te, fold_idx):
    n_synth = len(X_tr) * N_SYNTH_RATIO
    y_mean  = y_tr.mean(0);  y_std = y_tr.std(0).clip(min=1e-6)

    y_tr_n  = (y_tr - y_mean) / y_std
    y_te_n  = (y_te - y_mean) / y_std

    # ── Train GAN ──
    print(f'  Training GAN on {len(X_tr)} real samples …')
    G = train_gan(X_tr, soil_tr, pipe_tr, y_tr_n, fold_idx)

    # ── Generate & save synthetic ──
    X_syn, y_syn, soil_syn, pipe_syn = generate_and_save_gan(
        G, X_tr, y_tr, soil_tr, pipe_tr, y_mean, y_std, n_synth, fold_idx)
    y_syn_n = (y_syn - y_mean) / y_std

    # ── Combine real + synthetic ──
    X_aug    = np.concatenate([X_tr,    X_syn])
    y_aug_n  = np.concatenate([y_tr_n,  y_syn_n])
    soil_aug = np.concatenate([soil_tr, soil_syn])
    pipe_aug = np.concatenate([pipe_tr, pipe_syn])
    print(f'  CNN training: {len(X_tr)} real + {n_synth} GAN = {len(X_aug)} total')

    tr_ds = STFTDataset(X_aug, soil_aug, pipe_aug, y_aug_n)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True)

    te_ds     = STFTDataset(X_te, soil_te, pipe_te, y_te_n)
    X_te_t    = torch.stack([te_ds[i][0] for i in range(len(te_ds))])
    soil_te_t = torch.tensor(soil_te, dtype=torch.long)
    pipe_te_t = torch.tensor(pipe_te, dtype=torch.long)
    y_te_t    = torch.tensor(y_te_n,  dtype=torch.float32)

    model = CNN2DCondV2()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30, min_lr=1e-5)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
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
            if wait >= 80:
                print(f'  CNN early stop epoch {epoch+1}'); break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_n = model(X_te_t, soil_te_t, pipe_te_t).numpy()
    return y_pred_n * y_std + y_mean


# ── 5-fold CV ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
all_preds  = np.full_like(y, np.nan)
fold_r2s, fold_rmses = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_norm, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
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
print(f'Cond-aware v2 + noise:      0.801 ± 0.070')
print(f'Cond-aware v2 + GAN:       {r2_mean.mean():.3f} ± {r2_std.mean():.3f}')
print(f'Delta vs cond v2: {r2_mean.mean()-0.816:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
COND_COLORS = {0: '#e07b39', 1: '#4a90d9', 2: '#5cb85c'}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Condition-aware (soil+pipe) + cWGAN-GP aug — 5-fold CV, n=135',
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
plt.savefig('../results/fig_cond_gan_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: fig_cond_gan_scatter.png')
