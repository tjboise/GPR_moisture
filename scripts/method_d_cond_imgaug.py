"""
Condition-aware 2D CNN (CA-CNN) + traditional image augmentation on STFT.
Augmentations applied online during training on the spectrogram image:
  - Horizontal shift (time-axis offset)
  - Random brightness / contrast (signal amplitude variation)
  - Gaussian blur (bandwidth/resolution variation)
  - Gaussian noise (electronic noise)
  - CLAHE (local contrast enhancement)
No GAN — pure geometry/photometric augmentation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.signal import stft as scipy_stft
import albumentations as A
import pathlib, time

ROOT   = pathlib.Path(__file__).resolve().parent.parent
DATA   = ROOT / 'data'
RES    = ROOT / 'results'

# ── STFT params ────────────────────────────────────────────────────────────
FS       = 1e9 / 0.099609
NPERSEG  = 32; NOVERLAP = 28; NFFT = 512; FREQ_MAX = 3e9

# ── Training ───────────────────────────────────────────────────────────────
BATCH    = 16
EPOCHS   = 300
LR       = 1e-3
PATIENCE = 80
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP = {0: 0, 1: 1, 2: 1}

# ── Augmentation pipeline (training only) ─────────────────────────────────
TRAIN_AUG = A.Compose([
    # Time-axis shift: simulates slight arrival-time offset
    A.ShiftScaleRotate(
        shift_limit_x=0.08, shift_limit_y=0.0,
        scale_limit=0.0, rotate_limit=0, p=0.5),
    # Amplitude variation: different soil-antenna coupling strength
    A.RandomBrightnessContrast(
        brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    # Bandwidth / resolution variation
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    # Electronic noise
    A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
    # Local contrast enhancement (helps CNN find subtle features)
    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
])


def make_stft(trace):
    f, _, Zxx = scipy_stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                            nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)   # (H, W) in [0,1]


# ── Dataset ────────────────────────────────────────────────────────────────
class GPRDataset(Dataset):
    def __init__(self, X, y, cond, augment=False):
        self.imgs    = [make_stft(x) for x in X]
        self.y       = torch.tensor(y,    dtype=torch.float32)
        self.soil    = torch.tensor([SOIL_MAP[c] for c in cond], dtype=torch.long)
        self.pipe    = torch.tensor([PIPE_MAP[c] for c in cond], dtype=torch.long)
        self.augment = augment

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].copy()   # (H, W) float32 in [0,1]
        if self.augment:
            # albumentations expects HWC; add channel dim, then remove
            img_hwc = img[:, :, None]
            aug = TRAIN_AUG(image=img_hwc)['image']
            img = aug[:, :, 0]
            img = np.clip(img, 0.0, 1.0)
        return (torch.tensor(img).unsqueeze(0),   # (1, H, W)
                self.soil[i], self.pipe[i], self.y[i])


# ── Model (same CA-CNN as method_d_cond2.py) ──────────────────────────────
class CNN2DCondV2(nn.Module):
    def __init__(self, cond_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        feat_dim = 64 * 4 * 4 + cond_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x, soil, pipe):
        feat    = self.encoder(x).flatten(1)
        c_vec   = torch.cat([F.one_hot(soil, 2).float(),
                              F.one_hot(pipe, 2).float()], dim=1)
        return self.head(torch.cat([feat, c_vec], dim=1))


# ── 5-fold CV ──────────────────────────────────────────────────────────────
X    = np.load(DATA / 'X_norm.npy')
y    = np.load(DATA / 'y_moisture.npy')
cond = np.load(DATA / 'cond_labels.npy')

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y   = []; all_yhat = []; fold_r2 = []

print(f'Device: {DEVICE}')
print(f'\n{"Fold":>5}  {"R2_S":>7}  {"R2_T":>7}  {"R2_M":>7}  {"R2_B":>7}  {"Avg":>7}')

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, cond), 1):
    tr_dl = DataLoader(
        GPRDataset(X[tr_idx], y[tr_idx], cond[tr_idx], augment=True),
        batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(
        GPRDataset(X[va_idx], y[va_idx], cond[va_idx], augment=False),
        batch_size=BATCH)

    model     = CNN2DCondV2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=30, min_lr=1e-5)
    criterion = nn.MSELoss()

    best_val = float('inf'); best_wts = None; no_impr = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for imgs, soil, pipe, ys in tr_dl:
            imgs, soil, pipe, ys = (imgs.to(DEVICE), soil.to(DEVICE),
                                    pipe.to(DEVICE), ys.to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, soil, pipe), ys).backward()
            optimizer.step()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for imgs, soil, pipe, ys in va_dl:
                imgs, soil, pipe, ys = (imgs.to(DEVICE), soil.to(DEVICE),
                                        pipe.to(DEVICE), ys.to(DEVICE))
                val_loss += criterion(model(imgs, soil, pipe), ys).item() * len(imgs)
        val_loss /= len(va_idx)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_impr  = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE: break

    model.load_state_dict(best_wts); model.eval()
    preds = []
    with torch.no_grad():
        for imgs, soil, pipe, _ in va_dl:
            preds.append(model(imgs.to(DEVICE), soil.to(DEVICE),
                               pipe.to(DEVICE)).cpu().numpy())
    preds = np.vstack(preds)
    r2s   = [r2_score(y[va_idx][:, i], preds[:, i]) for i in range(4)]
    avg   = np.mean(r2s); fold_r2.append(avg)
    all_y.append(y[va_idx]); all_yhat.append(preds)
    print(f'{fold:>5}  {r2s[0]:>7.3f}  {r2s[1]:>7.3f}  {r2s[2]:>7.3f}  {r2s[3]:>7.3f}  {avg:>7.3f}')

all_y    = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
avg_r2   = np.mean(r2_final); std_r2 = np.std(fold_r2)

print(f'\n{"":5}  {"S":>7}  {"T":>7}  {"M":>7}  {"B":>7}  {"Avg":>7}')
print(f'{"Final":5}  {r2_final[0]:>7.3f}  {r2_final[1]:>7.3f}  '
      f'{r2_final[2]:>7.3f}  {r2_final[3]:>7.3f}  {avg_r2:>7.3f} ± {std_r2:.3f}')
print(f'\n=== 对比 ===')
print(f'Baseline (cond, 无 aug):       R² = 0.816')
print(f'+ cWGAN-GP aug:                R² = 0.825')
print(f'+ Image aug (本脚本):           R² = {avg_r2:.3f}')

# ── Scatter plot ───────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
labels = ['S (0 cm)', 'T (8 cm)', 'M (22 cm)', 'B (35 cm)']
for i, ax in enumerate(axes):
    ax.scatter(all_y[:, i], all_yhat[:, i], alpha=0.6, s=30,
               color='steelblue', edgecolors='k', lw=0.3)
    lo = min(all_y[:, i].min(), all_yhat[:, i].min()) - 1
    hi = max(all_y[:, i].max(), all_yhat[:, i].max()) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_title(f'{labels[i]}\nR²={r2_final[i]:.3f}', fontsize=10)
    ax.set_xlabel('Actual (%)'); ax.set_ylabel('Predicted (%)')
    ax.grid(True, alpha=0.3)
plt.suptitle(f'CA-CNN + Image Augmentation  |  Avg R² = {avg_r2:.3f}',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(RES / 'fig_imgaug_scatter.png', dpi=150, bbox_inches='tight')
print('Saved: fig_imgaug_scatter.png')
