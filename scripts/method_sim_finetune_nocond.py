"""
Sim pre-train + real fine-tune WITHOUT condition-aware head.
Tests whether pre-trained encoder alone (no soil/pipe metadata) is useful.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.signal import stft
import pathlib

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / 'data'
RES_DIR = ROOT / 'results'

FS = 1e9 / 0.099609
NPERSEG = 32; NOVERLAP = 28; NFFT = 512; FREQ_MAX = 3e9
BATCH = 16; EPOCHS = 200; LR_HEAD = 1e-3; LR_FULL = 2e-4
FREEZE_EPOCHS = 40; PATIENCE = 60
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_stft_image(trace):
    f, _, Zxx = stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)


class GPRDataset(Dataset):
    def __init__(self, X, y):
        self.imgs = [make_stft_image(x) for x in X]
        self.y    = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return torch.tensor(self.imgs[i]).unsqueeze(0), self.y[i]


class CNN2DFinetune(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.fc(self.encoder(x))

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True


X    = np.load(DATA / 'X_norm.npy')
y    = np.load(DATA / 'y_moisture.npy')
cond = np.load(DATA / 'cond_labels.npy')

encoder_path = RES_DIR / 'pretrained_encoder.pt'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y = []; all_yhat = []; fold_r2 = []

print(f'{"Fold":>5}  {"R2_S":>7}  {"R2_T":>7}  {"R2_M":>7}  {"R2_B":>7}  {"Avg":>7}')

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, cond), 1):
    tr_dl = DataLoader(GPRDataset(X[tr_idx], y[tr_idx]), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(GPRDataset(X[va_idx], y[va_idx]), batch_size=BATCH)

    model = CNN2DFinetune().to(DEVICE)
    model.encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))

    # Phase A: freeze encoder
    model.freeze_encoder()
    opt_a = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    crit  = nn.MSELoss()
    for _ in range(FREEZE_EPOCHS):
        model.train()
        for imgs, ys in tr_dl:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            opt_a.zero_grad(); crit(model(imgs), ys).backward(); opt_a.step()

    # Phase B: unfreeze all
    model.unfreeze_encoder()
    opt_b = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, factor=0.5, patience=20, min_lr=1e-5)
    best_val = float('inf'); best_wts = None; no_impr = 0

    for ep in range(1, EPOCHS - FREEZE_EPOCHS + 1):
        model.train()
        for imgs, ys in tr_dl:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            opt_b.zero_grad(); crit(model(imgs), ys).backward(); opt_b.step()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for imgs, ys in va_dl:
                imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
                val_loss += crit(model(imgs), ys).item() * len(imgs)
        val_loss /= len(va_idx); sched.step(val_loss)
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
        for imgs, ys in va_dl:
            preds.append(model(imgs.to(DEVICE)).cpu().numpy())
    preds = np.vstack(preds)
    r2s = [r2_score(y[va_idx][:, i], preds[:, i]) for i in range(4)]
    avg = np.mean(r2s); fold_r2.append(avg)
    all_y.append(y[va_idx]); all_yhat.append(preds)
    print(f'{fold:>5}  {r2s[0]:>7.3f}  {r2s[1]:>7.3f}  {r2s[2]:>7.3f}  {r2s[3]:>7.3f}  {avg:>7.3f}')

all_y = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
avg_r2   = np.mean(r2_final)

print(f'\n{"Final":5}  {r2_final[0]:.3f}  {r2_final[1]:.3f}  {r2_final[2]:.3f}  {r2_final[3]:.3f}  {avg_r2:.3f} ± {np.std(fold_r2):.3f}')
print(f'\n=== 对比 ===')
print(f'Baseline (无 condition, 无 sim):  R² = 0.797')
print(f'Sim pre-train + 无 condition:     R² = {avg_r2:.3f}')
print(f'Sim pre-train + condition head:   R² = 0.749')
print(f'Best (condition + cWGAN-GP):      R² = 0.825')
