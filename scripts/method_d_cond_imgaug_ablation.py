"""
Ablation: test each image augmentation technique individually on STFT + CA-CNN.
5 runs (one per augmentation type) × 5-fold CV each.
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
import pathlib, warnings
warnings.filterwarnings('ignore')

ROOT  = pathlib.Path(__file__).resolve().parent.parent
DATA  = ROOT / 'data'

FS = 1e9 / 0.099609
NPERSEG = 32; NOVERLAP = 28; NFFT = 512; FREQ_MAX = 3e9
BATCH = 16; EPOCHS = 300; LR = 1e-3; PATIENCE = 80
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP = {0: 0, 1: 1, 2: 1}

# ── 5 individual augmentation configs ─────────────────────────────────────
AUG_CONFIGS = {
    'HorizontalShift': A.Compose([
        A.Affine(translate_percent={'x': (-0.08, 0.08), 'y': (0, 0)},
                 scale=1.0, rotate=0, p=0.7),
    ]),
    'BrightnessContrast': A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    ]),
    'GaussianBlur': A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=0.7),
    ]),
    'GaussianNoise': A.Compose([
        A.GaussNoise(std_range=(0.02, 0.08), p=0.7),
    ]),
    'CLAHE': A.Compose([
        A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.7),
    ]),
}


def make_stft(trace):
    f, _, Zxx = scipy_stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                            nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)


class GPRDataset(Dataset):
    def __init__(self, X, y, cond, aug=None):
        self.imgs = [make_stft(x) for x in X]
        self.y    = torch.tensor(y,    dtype=torch.float32)
        self.soil = torch.tensor([SOIL_MAP[c] for c in cond], dtype=torch.long)
        self.pipe = torch.tensor([PIPE_MAP[c] for c in cond], dtype=torch.long)
        self.aug  = aug

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].copy()
        if self.aug is not None:
            img = np.clip(
                self.aug(image=img[:, :, None])['image'][:, :, 0], 0.0, 1.0)
        return (torch.tensor(img).unsqueeze(0),
                self.soil[i], self.pipe[i], self.y[i])


class CNN2DCondV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4 + 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x, soil, pipe):
        feat  = self.encoder(x).flatten(1)
        c_vec = torch.cat([F.one_hot(soil, 2).float(),
                            F.one_hot(pipe, 2).float()], dim=1)
        return self.fc(torch.cat([feat, c_vec], dim=1))


def run_cv(aug_transform):
    X    = np.load(DATA / 'X_norm.npy')
    y    = np.load(DATA / 'y_moisture.npy')
    cond = np.load(DATA / 'cond_labels.npy')
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y = []; all_yhat = []; fold_r2 = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, cond), 1):
        tr_dl = DataLoader(
            GPRDataset(X[tr_idx], y[tr_idx], cond[tr_idx], aug=aug_transform),
            batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(
            GPRDataset(X[va_idx], y[va_idx], cond[va_idx], aug=None),
            batch_size=BATCH)

        model     = CNN2DCondV2().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=30, min_lr=1e-5)
        criterion = nn.MSELoss()
        best_val  = float('inf'); best_wts = None; no_impr = 0

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
        fold_r2.append(np.mean(r2s))
        all_y.append(y[va_idx]); all_yhat.append(preds)

    all_y    = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
    r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
    return r2_final, np.std(fold_r2)


# ── Run all ablations ──────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
print(f'\n{"Method":<22}  {"S":>7}  {"T":>7}  {"M":>7}  {"B":>7}  {"Avg R²":>8}  {"±std":>6}')
print('-' * 72)

# Baseline (no aug)
r2_base, std_base = run_cv(None)
avg_base = np.mean(r2_base)
print(f'{"No augmentation":<22}  {r2_base[0]:>7.3f}  {r2_base[1]:>7.3f}  '
      f'{r2_base[2]:>7.3f}  {r2_base[3]:>7.3f}  {avg_base:>8.3f}  {std_base:>6.3f}')

results = {}
for name, aug in AUG_CONFIGS.items():
    r2s, std = run_cv(aug)
    avg = np.mean(r2s)
    results[name] = (r2s, avg, std)
    marker = ' ◀ better' if avg > avg_base else ''
    print(f'{name:<22}  {r2s[0]:>7.3f}  {r2s[1]:>7.3f}  '
          f'{r2s[2]:>7.3f}  {r2s[3]:>7.3f}  {avg:>8.3f}  {std:>6.3f}{marker}')

print('\n(Reference: cWGAN-GP aug R² = 0.825)')
