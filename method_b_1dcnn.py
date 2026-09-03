"""
Method B: 1D CNN — raw aligned A-scan → moisture regression
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

X_tr = torch.tensor(X_norm[idx_tr, None, :], dtype=torch.float32)  # (N, 1, 256)
X_te = torch.tensor(X_norm[idx_te, None, :], dtype=torch.float32)
y_tr = torch.tensor(y[idx_tr], dtype=torch.float32)
y_te = torch.tensor(y[idx_te], dtype=torch.float32)

# Normalize targets per-layer to help training
y_mean = y_tr.mean(0)
y_std  = y_tr.std(0).clamp(min=1e-6)
y_tr_n = (y_tr - y_mean) / y_std
y_te_n = (y_te - y_mean) / y_std

train_ds = TensorDataset(X_tr, y_tr_n)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)


# ── Model ─────────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=16, padding=8),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8,  padding=4),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=4, padding=2),  nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


model = CNN1D()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# ── Training with early stopping ──────────────────────────────────────────
best_val_loss = float('inf')
best_state    = None
patience      = 50
wait          = 0
train_losses  = []
val_losses    = []

for epoch in range(500):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_tr)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_te), y_te_n).item()

    train_losses.append(ep_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(best_state)

# ── Evaluation ────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_n = model(X_te).numpy()

y_pred = y_pred_n * y_std.numpy() + y_mean.numpy()
y_actual = y[idx_te]

print(f'\n{"Layer":<5} {"R²":>6} {"RMSE":>7}')
print('-' * 22)
r2s, rmses = [], []
for li, lbl in enumerate(LAYERS):
    r2   = r2_score(y_actual[:, li], y_pred[:, li])
    rmse = np.sqrt(mean_squared_error(y_actual[:, li], y_pred[:, li]))
    r2s.append(r2); rmses.append(rmse)
    print(f'{lbl:<5} {r2:>6.3f} {rmse:>7.2f}%')

# ── Plot 1: Training curve ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='Train loss', lw=1.5)
ax.plot(val_losses,   label='Val loss',   lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss (normalized)')
ax.set_title('1D CNN — Training curve', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodB_training.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Scatter plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Method B — 1D CNN: Predicted vs Actual moisture', fontsize=12, fontweight='bold')
for li, (ax, lbl) in enumerate(zip(axes, LAYERS)):
    ax.scatter(y_actual[:, li], y_pred[:, li], color='darkorange', edgecolors='white', s=60)
    mn = min(y_actual[:, li].min(), y_pred[:, li].min()) - 1
    mx = max(y_actual[:, li].max(), y_pred[:, li].max()) + 1
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_title(f'{lbl} layer\nR²={r2s[li]:.3f}, RMSE={rmses[li]:.2f}%', fontsize=10)
    ax.set_xlabel('Actual moisture (%)'); ax.set_ylabel('Predicted moisture (%)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_methodB_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

np.save('results_methodB.npy', {'r2': r2s, 'rmse': rmses}, allow_pickle=True)
print('\nSaved: fig_methodB_training.png, fig_methodB_scatter.png')
