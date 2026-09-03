"""
Method D: STFT (tuned parameters) + 2D CNN → moisture regression
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

LAYERS = ['S', 'T', 'M', 'B']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_norm  = np.load('X_norm.npy')       # (112, 256)
y       = np.load('y_moisture.npy')   # (112, 4)
idx_tr  = np.load('idx_train.npy')
idx_te  = np.load('idx_test.npy')

# ── STFT parameters (tuned for better resolution) ─────────────────────────
# dt = 0.099609 ns → fs ≈ 10.04 GHz
FS_HZ    = 1e9 / 0.099609
NPERSEG  = 32          # smaller window → better time resolution (~3.2 ns)
NOVERLAP = 28          # 87.5% overlap → smooth time axis
NFFT     = 512         # zero-pad → finer frequency grid

def make_stft_image(trace):
    f, t, Zxx = stft(trace, fs=FS_HZ, nperseg=NPERSEG, noverlap=NOVERLAP,
                     nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)
    # Keep 0–3 GHz (physically meaningful range)
    f_mask = f <= 3e9
    img = mag[f_mask, :]                         # (freq_bins, time_bins)
    img = img / (img.max() + 1e-12)              # normalize to [0,1]
    return img.astype(np.float32)

# Build STFT image dataset
sample_img = make_stft_image(X_norm[0])
H, W = sample_img.shape
print(f'STFT image shape per sample: {H} × {W}  (freq × time)')

imgs = np.stack([make_stft_image(X_norm[i]) for i in range(len(X_norm))])
# Shape: (112, H, W) → add channel dim → (112, 1, H, W)
imgs = imgs[:, None, :, :]

X_tr_t = torch.tensor(imgs[idx_tr], dtype=torch.float32)
X_te_t = torch.tensor(imgs[idx_te], dtype=torch.float32)
y_tr_t = torch.tensor(y[idx_tr],    dtype=torch.float32)
y_te_t = torch.tensor(y[idx_te],    dtype=torch.float32)

y_mean = y_tr_t.mean(0); y_std = y_tr_t.std(0).clamp(min=1e-6)
y_tr_n = (y_tr_t - y_mean) / y_std
y_te_n = (y_te_t - y_mean) / y_std

train_ds = TensorDataset(X_tr_t, y_tr_n)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)


# ── 2D CNN model ──────────────────────────────────────────────────────────
class CNN2D(nn.Module):
    def __init__(self, in_h, in_w):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


model = CNN2D(H, W)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state    = None
patience = 50; wait = 0
train_losses = []; val_losses = []

for epoch in range(500):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_tr_t)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_te_t), y_te_n).item()

    train_losses.append(ep_loss); val_losses.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    y_pred_n = model(X_te_t).numpy()

y_pred   = y_pred_n * y_std.numpy() + y_mean.numpy()
y_actual = y[idx_te]

print(f'\n{"Layer":<5} {"R²":>6} {"RMSE":>7}')
print('-' * 22)
r2s, rmses = [], []
for li, lbl in enumerate(LAYERS):
    r2   = r2_score(y_actual[:, li], y_pred[:, li])
    rmse = np.sqrt(mean_squared_error(y_actual[:, li], y_pred[:, li]))
    r2s.append(r2); rmses.append(rmse)
    print(f'{lbl:<5} {r2:>6.3f} {rmse:>7.2f}%')

# ── Plot: training curve ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='Train'); ax.plot(val_losses, label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss (normalized)')
ax.set_title('Method D — STFT + 2D CNN: Training curve', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodD_training.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot: scatter ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Method D — STFT + 2D CNN: Predicted vs Actual', fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y_actual[:, li], y_pred[:, li], color='purple', edgecolors='white', s=60)
    mn = min(y_actual[:, li].min(), y_pred[:, li].min()) - 1
    mx = max(y_actual[:, li].max(), y_pred[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl} layer\nR²={r2s[li]:.3f}, RMSE={rmses[li]:.2f}%', fontsize=10)
    ax.set_xlabel('Actual moisture (%)'); ax.set_ylabel('Predicted moisture (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodD_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Show example STFT images (tuned params) ───────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle(f'STFT images (nperseg={NPERSEG}, noverlap={NOVERLAP}, nfft={NFFT})\n'
             'Top: low moisture examples | Bottom: high moisture examples', fontsize=11, fontweight='bold')
b_mo = y[:, 3]
sorted_idx = np.argsort(b_mo)
for col, idx in enumerate(sorted_idx[:4]):
    ax = axes[0, col]
    ax.imshow(imgs[idx, 0], aspect='auto', origin='lower', cmap='hot',
              extent=[0, imgs.shape[-1], 0, H])
    ax.set_title(f'B-moist={b_mo[idx]:.1f}%', fontsize=9)
    ax.set_xlabel('Time bins'); ax.set_ylabel('Freq bins')
for col, idx in enumerate(sorted_idx[-4:]):
    ax = axes[1, col]
    ax.imshow(imgs[idx, 0], aspect='auto', origin='lower', cmap='hot',
              extent=[0, imgs.shape[-1], 0, H])
    ax.set_title(f'B-moist={b_mo[idx]:.1f}%', fontsize=9)
    ax.set_xlabel('Time bins'); ax.set_ylabel('Freq bins')
plt.tight_layout()
plt.savefig('fig_methodD_stft_samples.png', dpi=150, bbox_inches='tight')
plt.close()

np.save('results_methodD.npy', {'r2': r2s, 'rmse': rmses}, allow_pickle=True)
print('\nSaved: fig_methodD_training.png, fig_methodD_scatter.png, fig_methodD_stft_samples.png')
