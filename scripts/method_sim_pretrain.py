"""
Stage 1: Pre-train CNN encoder on gprMax simulation data (125 samples, 1.2 GHz).
No condition-aware head — encoder learns general GPR→moisture mapping.

Saves: results/pretrained_encoder.pt  (encoder weights only)
       results/pretrained_full.pt      (full model, for inspection)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import stft
import pathlib, time

ROOT    = pathlib.Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / 'data' / 'sim'
RES_DIR = ROOT / 'results'

# ── STFT params (same as real-data pipeline) ───────────────────────────────
FS        = 1e9 / 0.099609   # nominal sampling frequency
NPERSEG   = 32
NOVERLAP  = 28
NFFT      = 512
FREQ_MAX  = 3e9

# ── Training config ────────────────────────────────────────────────────────
BATCH       = 16
EPOCHS      = 300
LR          = 1e-3
VAL_SPLIT   = 0.15    # ~19 samples held out for validation
PATIENCE    = 50
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Dataset ────────────────────────────────────────────────────────────────
def make_stft_image(trace):
    f, _, Zxx = stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)


class SimDataset(Dataset):
    def __init__(self, X, y):
        self.imgs = [make_stft_image(x) for x in X]
        self.y    = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = torch.tensor(self.imgs[i]).unsqueeze(0)   # (1, H, W)
        return img, self.y[i]


# ── Model (encoder only, no condition head) ────────────────────────────────
class CNN2DPretrain(nn.Module):
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Load data ──────────────────────────────────────────────────────────────
X = np.load(SIM_DIR / 'X_sim.npy')   # (125, 256)
y = np.load(SIM_DIR / 'y_sim.npy')   # (125, 4)

print(f'Sim data: {X.shape}  moisture range {y.min():.1f}–{y.max():.1f}%')
print(f'Device: {DEVICE}')

dataset  = SimDataset(X, y)
n_val    = max(1, int(len(dataset) * VAL_SPLIT))
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)

print(f'Train: {n_train}  Val: {n_val}')

# ── Train ──────────────────────────────────────────────────────────────────
model     = CNN2DPretrain().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=20, min_lr=1e-5)
criterion = nn.MSELoss()

best_val_loss = float('inf')
no_improve    = 0
t0            = time.time()

print(f'\n{"Epoch":>6}  {"Train MSE":>10}  {"Val MSE":>10}  {"LR":>8}')
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, ys in train_dl:
        imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), ys)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(imgs)
    train_loss /= n_train

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, ys in val_dl:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            val_loss += criterion(model(imgs), ys).item() * len(imgs)
    val_loss /= n_val
    scheduler.step(val_loss)

    if epoch % 50 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f'{epoch:>6}  {train_loss:>10.3f}  {val_loss:>10.3f}  {lr_now:>8.2e}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), RES_DIR / 'pretrained_full.pt')
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

elapsed = time.time() - t0
print(f'\nBest val MSE: {best_val_loss:.3f}  ({elapsed:.0f}s)')

# Save encoder weights separately for fine-tuning
model.load_state_dict(torch.load(RES_DIR / 'pretrained_full.pt', map_location=DEVICE))
torch.save(model.encoder.state_dict(), RES_DIR / 'pretrained_encoder.pt')
print('Saved: pretrained_encoder.pt  pretrained_full.pt')
