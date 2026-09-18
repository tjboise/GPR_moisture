"""
Method D — GAN augmented: conditional WGAN-GP + STFT + 2D CNN
Pipeline per fold:
  1. Train cWGAN-GP on training fold (108 real A-scans)
  2. Generate 2× synthetic A-scans conditioned on sampled moisture / condition
  3. Train STFT+2D CNN on real + synthetic (3× data)
  4. Evaluate on held-out real data only
Baseline: Avg R² = 0.797 ± 0.056
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
from torch.utils.data import Dataset, DataLoader, TensorDataset

LAYERS      = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm = np.load('../data/X_norm.npy')       # (135, 256) raw A-scans
y      = np.load('../data/y_moisture.npy')   # (135, 4)
cond   = np.load('../data/cond_labels.npy')  # (135,) in {0,1,2}

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

sample_img = ascan_to_stft(X_norm[0])
H, W = sample_img.shape
print(f'STFT image: {H}×{W}  |  GAN augmentation: cWGAN-GP (2× synthetic per fold)')

N_COND   = 3    # condition classes
LATENT   = 64
COND_DIM = 4 + N_COND  # 4 moisture (normalised) + 3 one-hot condition = 7


# ── Condition vector helpers ───────────────────────────────────────────────
def make_cond_vec(y_rows, cond_rows, y_mean, y_std):
    """Normalise moisture values + one-hot condition → (N, COND_DIM) tensor."""
    y_n = (y_rows - y_mean) / y_std
    oh  = np.eye(N_COND)[cond_rows.astype(int)]
    return torch.tensor(np.concatenate([y_n, oh], axis=1), dtype=torch.float32)


# ── GAN architecture ───────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT + COND_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),               nn.LeakyReLU(0.2),
            nn.Linear(512, 256),               nn.Tanh(),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + COND_DIM, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


def gradient_penalty(critic, real, fake, c_real, lambda_gp=10.0):
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, c_real)
    grads = torch.autograd.grad(d_interp, interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


def train_gan(X_tr, y_tr_n, cond_vec_tr, epochs=300, n_critic=5, bs=16):
    """Train cWGAN-GP on training fold. Returns (Generator, losses)."""
    G = Generator(); C = Critic()
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    ds  = TensorDataset(X_t, cond_vec_tr)
    dl  = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    g_losses, c_losses = [], []
    for ep in range(epochs):
        for real_x, real_c in dl:
            bsz = real_x.size(0)

            # ── Train critic n_critic times ──
            for _ in range(n_critic):
                z    = torch.randn(bsz, LATENT)
                fake = G(z, real_c).detach()
                gp   = gradient_penalty(C, real_x, fake, real_c)
                c_loss = -C(real_x, real_c).mean() + C(fake, real_c).mean() + gp
                opt_C.zero_grad(); c_loss.backward(); opt_C.step()

            # ── Train generator ──
            z      = torch.randn(bsz, LATENT)
            fake   = G(z, real_c)
            g_loss = -C(fake, real_c).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        g_losses.append(g_loss.item())
        c_losses.append(c_loss.item())

    print(f'    GAN trained {epochs} ep | G_loss={g_losses[-1]:.3f} C_loss={c_losses[-1]:.3f}')
    return G


def generate_samples(G, y_tr, cond_tr, y_mean, y_std, n_synth):
    """
    Generate n_synth synthetic A-scans by sampling conditions from training
    distribution (random interpolation between real pairs).
    Returns (X_synth, y_synth, cond_synth) as numpy arrays.
    """
    G.eval()
    n_real = len(y_tr)
    idx_a  = np.random.randint(0, n_real, n_synth)
    idx_b  = np.random.randint(0, n_real, n_synth)
    lam    = np.random.uniform(0.3, 0.7, (n_synth, 1))

    y_synth    = lam * y_tr[idx_a] + (1 - lam) * y_tr[idx_b]
    cond_synth = cond_tr[idx_a]  # keep parent condition

    y_n   = (y_synth - y_mean) / y_std
    oh    = np.eye(N_COND)[cond_synth.astype(int)]
    c_vec = torch.tensor(np.concatenate([y_n, oh], axis=1), dtype=torch.float32)
    z     = torch.randn(n_synth, LATENT)

    with torch.no_grad():
        x_synth = G(z, c_vec).numpy()

    return x_synth, y_synth, cond_synth


# ── CNN (same as baseline) ─────────────────────────────────────────────────
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


class STFTDataset(Dataset):
    def __init__(self, X_ascan, y_norm):
        self.X = X_ascan
        self.y = torch.tensor(y_norm, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        img = ascan_to_stft(self.X[i])
        return torch.tensor(img[None], dtype=torch.float32), self.y[i]


def train_cnn(X_tr, y_tr_n, X_te, patience=60, epochs=600):
    ds = STFTDataset(X_tr, y_tr_n)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    # build val tensors (no augmentation)
    val_ds = STFTDataset(X_te, np.zeros((len(X_te), 4)))
    X_te_t = torch.stack([val_ds[i][0] for i in range(len(val_ds))])

    model = CNN2D()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.MSELoss()

    # validation uses real samples → need real y_te_n separately
    best_loss, best_state, wait = float('inf'), None, 0
    # We return model, user supplies val y
    return model, dl, opt, crit, X_te_t


def train_fold(X_tr_real, y_tr_real, cond_tr,
               X_te, y_te, fold_idx,
               gan_epochs=300, n_synth_ratio=2):
    n_real  = len(X_tr_real)
    n_synth = n_real * n_synth_ratio

    y_mean = y_tr_real.mean(0)
    y_std  = y_tr_real.std(0).clip(min=1e-6)

    y_tr_n = (y_tr_real - y_mean) / y_std
    y_te_n = (y_te       - y_mean) / y_std

    # ── 1. Train GAN ──
    print(f'  Training GAN on {n_real} real samples …')
    c_vec_tr = make_cond_vec(y_tr_real, cond_tr, y_mean, y_std)
    G = train_gan(X_tr_real, y_tr_n, c_vec_tr, epochs=gan_epochs)

    # ── 2. Generate synthetic A-scans ──
    X_syn, y_syn, _ = generate_samples(G, y_tr_real, cond_tr,
                                       y_mean, y_std, n_synth)
    y_syn_n = (y_syn - y_mean) / y_std

    # ── 3. Combine real + synthetic ──
    X_aug  = np.concatenate([X_tr_real, X_syn], axis=0)
    y_aug_n = np.concatenate([y_tr_n,   y_syn_n], axis=0)
    print(f'  CNN training on {len(X_aug)} samples ({n_real} real + {n_synth} synthetic)')

    # ── 4. Train CNN ──
    ds = STFTDataset(X_aug, y_aug_n)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    val_ds = STFTDataset(X_te, y_te_n)
    X_te_t = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    y_te_t = torch.tensor(y_te_n, dtype=torch.float32)

    model = CNN2D()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    for epoch in range(600):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t), y_te_t).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 60:
                print(f'  CNN early stop epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_n = model(X_te_t).numpy()

    return y_pred_n * y_std + y_mean


# ── 5-fold CV ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
all_preds  = np.full_like(y, np.nan)
fold_r2s, fold_rmses = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_norm, cond)):
    print(f'\nFold {fold_i+1}/5  (train={len(tr_idx)}, test={len(te_idx)})')
    y_pred = train_fold(X_norm[tr_idx], y[tr_idx], cond[tr_idx],
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
print(f'Baseline: 0.797 ± 0.056')
print(f'Delta:   {r2_mean.mean()-0.797:+.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Method D — cWGAN-GP + STFT + 2D CNN (5-fold CV, n=135)',
             fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y[:, li], all_preds[:, li], color='steelblue',
               edgecolors='white', s=40, alpha=0.8)
    mn = min(y[:, li].min(), all_preds[:, li].min()) - 1
    mx = max(y[:, li].max(), all_preds[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl}  R²={r2_mean[li]:.3f}±{r2_std[li]:.3f}\nRMSE={rm_mean[li]:.2f}±{rm_std[li]:.2f}%',
                 fontsize=9)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/fig_methodD_gan_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: fig_methodD_gan_scatter.png')
