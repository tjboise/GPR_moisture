"""
Raw A-scan + 1D CNN baseline (no STFT).
Two variants:
  (a) Plain 1D CNN       — no condition input
  (b) Condition-aware 1D CNN (CA-1DCNN) — soil + pipe one-hot

Used to verify that STFT representation is necessary:
  if 2D CNN on STFT >> 1D CNN on raw A-scan, the time-frequency
  representation adds genuine value beyond what the raw signal provides.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
import pathlib

ROOT  = pathlib.Path(__file__).resolve().parent.parent
DATA  = ROOT / 'data'

BATCH    = 16
EPOCHS   = 300
LR       = 1e-3
PATIENCE = 80
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP = {0: 0, 1: 1, 2: 1}


# ── Dataset ────────────────────────────────────────────────────────────────

class AScanDataset(Dataset):
    def __init__(self, X, y, cond=None):
        self.X    = torch.tensor(X, dtype=torch.float32)          # (N, 256)
        self.y    = torch.tensor(y, dtype=torch.float32)
        self.soil = (torch.tensor([SOIL_MAP[c] for c in cond], dtype=torch.long)
                     if cond is not None else None)
        self.pipe = (torch.tensor([PIPE_MAP[c] for c in cond], dtype=torch.long)
                     if cond is not None else None)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].unsqueeze(0)      # (1, 256)  channel-first for Conv1d
        if self.soil is not None:
            return x, self.soil[i], self.pipe[i], self.y[i]
        return x, self.y[i]


# ── Models ─────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """Plain 1D CNN, no condition."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x): return self.fc(self.encoder(x))


class CNN1DCond(nn.Module):
    """Condition-aware 1D CNN (soil + pipe one-hot)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 + 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x, soil, pipe):
        feat  = self.encoder(x).flatten(1)
        c_vec = torch.cat([F.one_hot(soil, 2).float(),
                           F.one_hot(pipe, 2).float()], dim=1)
        return self.fc(torch.cat([feat, c_vec], dim=1))


# ── CV helpers ─────────────────────────────────────────────────────────────

def run_cv_plain(X, y, cond):
    """5-fold CV for plain 1D CNN (no condition)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y = []; all_yhat = []; fold_r2 = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, cond), 1):
        tr_dl = DataLoader(AScanDataset(X[tr_idx], y[tr_idx]),
                           batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(AScanDataset(X[va_idx], y[va_idx]),
                           batch_size=BATCH)

        model = CNN1D().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=30, min_lr=1e-5)
        crit  = nn.MSELoss()
        best_val = float('inf'); best_wts = None; no_impr = 0

        for _ in range(EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_dl:
                    val_loss += crit(model(xb.to(DEVICE)),
                                    yb.to(DEVICE)).item() * len(xb)
            val_loss /= len(va_idx); sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_impr = 0
            else:
                no_impr += 1
                if no_impr >= PATIENCE: break

        model.load_state_dict(best_wts); model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in va_dl:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
        preds = np.vstack(preds)
        r2s   = [r2_score(y[va_idx][:, i], preds[:, i]) for i in range(4)]
        fold_r2.append(np.mean(r2s))
        all_y.append(y[va_idx]); all_yhat.append(preds)

    all_y    = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
    r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
    rmse     = [np.sqrt(mean_squared_error(all_y[:, i], all_yhat[:, i])) for i in range(4)]
    return r2_final, rmse, np.std(fold_r2)


def run_cv_cond(X, y, cond):
    """5-fold CV for condition-aware 1D CNN."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y = []; all_yhat = []; fold_r2 = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, cond), 1):
        tr_dl = DataLoader(AScanDataset(X[tr_idx], y[tr_idx], cond[tr_idx]),
                           batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(AScanDataset(X[va_idx], y[va_idx], cond[va_idx]),
                           batch_size=BATCH)

        model = CNN1DCond().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=30, min_lr=1e-5)
        crit  = nn.MSELoss()
        best_val = float('inf'); best_wts = None; no_impr = 0

        for _ in range(EPOCHS):
            model.train()
            for xb, sb, pb, yb in tr_dl:
                xb, sb, pb, yb = (xb.to(DEVICE), sb.to(DEVICE),
                                   pb.to(DEVICE), yb.to(DEVICE))
                opt.zero_grad(); crit(model(xb, sb, pb), yb).backward(); opt.step()
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for xb, sb, pb, yb in va_dl:
                    val_loss += crit(model(xb.to(DEVICE), sb.to(DEVICE),
                                          pb.to(DEVICE)),
                                    yb.to(DEVICE)).item() * len(xb)
            val_loss /= len(va_idx); sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_impr = 0
            else:
                no_impr += 1
                if no_impr >= PATIENCE: break

        model.load_state_dict(best_wts); model.eval()
        preds = []
        with torch.no_grad():
            for xb, sb, pb, _ in va_dl:
                preds.append(model(xb.to(DEVICE), sb.to(DEVICE),
                                   pb.to(DEVICE)).cpu().numpy())
        preds = np.vstack(preds)
        r2s   = [r2_score(y[va_idx][:, i], preds[:, i]) for i in range(4)]
        fold_r2.append(np.mean(r2s))
        all_y.append(y[va_idx]); all_yhat.append(preds)

    all_y    = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
    r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
    rmse     = [np.sqrt(mean_squared_error(all_y[:, i], all_yhat[:, i])) for i in range(4)]
    return r2_final, rmse, np.std(fold_r2)


# ── Main ───────────────────────────────────────────────────────────────────

X    = np.load(DATA / 'X_norm.npy')
y    = np.load(DATA / 'y_moisture.npy')
cond = np.load(DATA / 'cond_labels.npy')

print(f'Device: {DEVICE}')
print(f'Data: X={X.shape}  y={y.shape}\n')

print('Running plain 1D CNN (no condition)...')
r2_plain, rmse_plain, std_plain = run_cv_plain(X, y, cond)

print('Running condition-aware 1D CNN...')
r2_cond, rmse_cond, std_cond = run_cv_cond(X, y, cond)

# ── Summary table ──────────────────────────────────────────────────────────
layers = ['S', 'T', 'M', 'B']
print(f'\n{"Model":<30}  {"S":>6}  {"T":>6}  {"M":>6}  {"B":>6}  {"Avg R2":>8}  {"+-std":>6}')
print('-' * 78)

def fmt(r2s, std):
    return (f'  {r2s[0]:>6.3f}  {r2s[1]:>6.3f}  {r2s[2]:>6.3f}  {r2s[3]:>6.3f}'
            f'  {np.mean(r2s):>8.3f}  {std:>6.3f}')

print(f'{"1D CNN (raw A-scan)":<30}' + fmt(r2_plain, std_plain))
print(f'{"CA-1D CNN (raw A-scan)":<30}' + fmt(r2_cond, std_cond))
print()
print(f'{"--- reference (STFT-based) ---":<30}')
print(f'{"STFT + 2D CNN":<30}   0.870   0.754   0.743   0.820     0.797   0.056')
print(f'{"STFT + CA-2D CNN":<30}   0.853   0.800   0.781   0.828     0.816   0.056')
print(f'{"STFT + CA-2D CNN + cWGAN-GP":<30}   0.884   0.764   0.811   0.843     0.825   0.056')

print(f'\nSTFT advantage over raw A-scan:')
print(f'  vs plain CNN:  {0.797 - np.mean(r2_plain):+.3f} (baseline)')
print(f'  vs CA-CNN:     {0.816 - np.mean(r2_cond):+.3f} (with condition)')
