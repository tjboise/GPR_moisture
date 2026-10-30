"""
Stage 2: Fine-tune pre-trained encoder on real lab data with condition-aware head.
Compares against cond2 baseline (no pre-training) under same 5-fold CV.

Strategy:
  Phase A (epochs 1-40):  freeze encoder, train head only  → head adapts to real data
  Phase B (epoch 41+):    unfreeze all, small LR            → full fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.signal import stft
import pathlib, time

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / 'data'
RES_DIR = ROOT / 'results'

# ── STFT / training params ─────────────────────────────────────────────────
FS       = 1e9 / 0.099609
NPERSEG  = 32; NOVERLAP = 28; NFFT = 512; FREQ_MAX = 3e9
BATCH    = 16; EPOCHS = 200; LR_HEAD = 1e-3; LR_FULL = 2e-4
FREEZE_EPOCHS = 40; PATIENCE = 60
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP = {0: 0, 1: 1, 2: 1}


# ── Dataset ────────────────────────────────────────────────────────────────
def make_stft_image(trace):
    f, _, Zxx = stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)


class GPRDataset(Dataset):
    def __init__(self, X, y, cond):
        self.imgs  = [make_stft_image(x) for x in X]
        self.y     = torch.tensor(y,    dtype=torch.float32)
        self.soil  = torch.tensor([SOIL_MAP[c] for c in cond], dtype=torch.long)
        self.pipe  = torch.tensor([PIPE_MAP[c] for c in cond], dtype=torch.long)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return (torch.tensor(self.imgs[i]).unsqueeze(0),
                self.soil[i], self.pipe[i], self.y[i])


# ── Model ──────────────────────────────────────────────────────────────────
class CNN2DCondFinetune(nn.Module):
    def __init__(self, cond_dim=4):
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
            nn.Flatten(),                          # placeholder — rebuilt in forward
        )
        feat_dim = 64 * 4 * 4 + cond_dim
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, x, soil, pipe):
        feat    = self.encoder(x).flatten(1)
        soil_oh = F.one_hot(soil, num_classes=2).float()
        pipe_oh = F.one_hot(pipe, num_classes=2).float()
        c_vec   = torch.cat([soil_oh, pipe_oh], dim=1)
        return self.fc(torch.cat([feat, c_vec], dim=1))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# ── Load real data ─────────────────────────────────────────────────────────
X    = np.load(DATA / 'X_norm.npy')
y    = np.load(DATA / 'y_moisture.npy')
cond = np.load(DATA / 'cond_labels.npy')
print(f'Real data: {X.shape}  Device: {DEVICE}')

encoder_path = RES_DIR / 'pretrained_encoder.pt'
if not encoder_path.exists():
    raise FileNotFoundError('Run method_sim_pretrain.py first to generate pretrained_encoder.pt')

# ── 5-fold CV ──────────────────────────────────────────────────────────────
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y   = []; all_yhat = []
fold_r2 = []

print(f'\n{"Fold":>5}  {"R2_S":>7}  {"R2_T":>7}  {"R2_M":>7}  {"R2_B":>7}  {"Avg":>7}')

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, cond), 1):
    X_tr, y_tr, c_tr = X[tr_idx], y[tr_idx], cond[tr_idx]
    X_va, y_va, c_va = X[val_idx], y[val_idx], cond[val_idx]

    tr_dl = DataLoader(GPRDataset(X_tr, y_tr, c_tr), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(GPRDataset(X_va, y_va, c_va), batch_size=BATCH)

    # Build model and load pre-trained encoder
    model = CNN2DCondFinetune().to(DEVICE)
    model.encoder.load_state_dict(
        torch.load(encoder_path, map_location=DEVICE))

    # ── Phase A: freeze encoder, train head only ──────────────────────────
    model.freeze_encoder()
    opt_a = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    crit = nn.MSELoss()

    for ep in range(1, FREEZE_EPOCHS + 1):
        model.train()
        for imgs, soil, pipe, ys in tr_dl:
            imgs, soil, pipe, ys = imgs.to(DEVICE), soil.to(DEVICE), pipe.to(DEVICE), ys.to(DEVICE)
            opt_a.zero_grad()
            crit(model(imgs, soil, pipe), ys).backward()
            opt_a.step()

    # ── Phase B: unfreeze all, fine-tune with small LR ────────────────────
    model.unfreeze_encoder()
    opt_b = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_b, factor=0.5, patience=20, min_lr=1e-5)

    best_val  = float('inf')
    best_wts  = None
    no_impr   = 0

    for ep in range(1, EPOCHS - FREEZE_EPOCHS + 1):
        model.train()
        for imgs, soil, pipe, ys in tr_dl:
            imgs, soil, pipe, ys = imgs.to(DEVICE), soil.to(DEVICE), pipe.to(DEVICE), ys.to(DEVICE)
            opt_b.zero_grad()
            crit(model(imgs, soil, pipe), ys).backward()
            opt_b.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, soil, pipe, ys in va_dl:
                imgs, soil, pipe, ys = imgs.to(DEVICE), soil.to(DEVICE), pipe.to(DEVICE), ys.to(DEVICE)
                val_loss += crit(model(imgs, soil, pipe), ys).item() * len(imgs)
        val_loss /= len(val_idx)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_impr  = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE:
                break

    # ── Evaluate on validation fold ───────────────────────────────────────
    model.load_state_dict(best_wts)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, soil, pipe, ys in va_dl:
            imgs, soil, pipe = imgs.to(DEVICE), soil.to(DEVICE), pipe.to(DEVICE)
            preds.append(model(imgs, soil, pipe).cpu().numpy())
    preds = np.vstack(preds)

    r2s = [r2_score(y_va[:, i], preds[:, i]) for i in range(4)]
    avg = np.mean(r2s)
    fold_r2.append(avg)
    all_y.append(y_va); all_yhat.append(preds)
    print(f'{fold:>5}  {r2s[0]:>7.3f}  {r2s[1]:>7.3f}  {r2s[2]:>7.3f}  {r2s[3]:>7.3f}  {avg:>7.3f}')

# ── Overall results ────────────────────────────────────────────────────────
all_y    = np.vstack(all_y)
all_yhat = np.vstack(all_yhat)
r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
avg_r2   = np.mean(r2_final)
std_r2   = np.std(fold_r2)

print(f'\n{"":5}  {"S":>7}  {"T":>7}  {"M":>7}  {"B":>7}  {"Avg":>7}')
print(f'{"Final":5}  {r2_final[0]:>7.3f}  {r2_final[1]:>7.3f}  '
      f'{r2_final[2]:>7.3f}  {r2_final[3]:>7.3f}  {avg_r2:>7.3f} ± {std_r2:.3f}')
print(f'\nBaseline (cond+GAN): Avg R² = 0.825')
print(f'Sim pre-train+fine-tune: Avg R² = {avg_r2:.3f}  ({"+" if avg_r2>0.825 else ""}{avg_r2-0.825:.3f})')

# ── Save scatter plot ──────────────────────────────────────────────────────
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
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Sim pre-train + Real fine-tune  |  Avg R² = {avg_r2:.3f}',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(RES_DIR / 'fig_sim_finetune_scatter.png', dpi=150, bbox_inches='tight')
print('Saved: fig_sim_finetune_scatter.png')
