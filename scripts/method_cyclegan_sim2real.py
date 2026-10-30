"""
CycleGAN domain adaptation: sim A-scans → real-style A-scans.

Pipeline:
  1. Train CycleGAN (G_S2R, G_R2S) on unpaired sim (375) + real (135) A-scans
  2. Translate all 375 sim A-scans to real domain: X_trans = G_S2R(X_sim)
  3. 5-fold CV on real data:
       train = real_train_fold (108) + X_trans (375) with sim moisture labels
       val   = real_val_fold  (27)  — real only, fair evaluation

CycleGAN operates on raw A-scans (1D signals, 256 samples).
STFT conversion happens after translation, inside the CNN Dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.signal import stft as scipy_stft
import pathlib, itertools

ROOT   = pathlib.Path(__file__).resolve().parent.parent
DATA   = ROOT / 'data'
SIM    = DATA / 'sim'
RES    = ROOT / 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── CycleGAN hyperparams ──────────────────────────────────────────────────
CYC_EPOCHS   = 200
CYC_LR       = 2e-4
LAMBDA_CYC   = 10.0
LAMBDA_ID    = 5.0
CYC_BATCH    = 16

# ── CNN finetune hyperparams ──────────────────────────────────────────────
FS = 1e9 / 0.099609
NPERSEG = 32; NOVERLAP = 28; NFFT = 512; FREQ_MAX = 3e9
BATCH = 16; EPOCHS = 300; LR = 1e-3; PATIENCE = 80
SOIL_MAP = {0: 0, 1: 0, 2: 1}
PIPE_MAP  = {0: 0, 1: 1, 2: 1}


# ═══════════════════════════════════════════════════════════════════════════
# CycleGAN Components (1D)
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm1d(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm1d(ch),
        )
    def forward(self, x): return x + self.net(x)


class Generator1D(nn.Module):
    """Encoder-ResBlock-Decoder. Input/output: (B, 1, 256) in [-1, 1]."""
    def __init__(self, n_res=6):
        super().__init__()
        layers = [
            nn.Conv1d(1, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm1d(256), nn.ReLU(inplace=True),
        ]
        layers += [ResBlock1D(256) for _ in range(n_res)]
        layers += [
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, 7, padding=3, padding_mode='reflect'),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


class Discriminator1D(nn.Module):
    """PatchGAN for 1D signals."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1, 4, padding=1),
        )
    def forward(self, x): return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════
# CycleGAN Training
# ═══════════════════════════════════════════════════════════════════════════

def train_cyclegan(X_sim, X_real):
    """Train CycleGAN on unpaired (sim, real) 1D A-scans. Returns G_S2R."""
    G_S2R = Generator1D().to(DEVICE)
    G_R2S = Generator1D().to(DEVICE)
    D_R   = Discriminator1D().to(DEVICE)
    D_S   = Discriminator1D().to(DEVICE)

    opt_G = torch.optim.Adam(
        itertools.chain(G_S2R.parameters(), G_R2S.parameters()),
        lr=CYC_LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        itertools.chain(D_R.parameters(), D_S.parameters()),
        lr=CYC_LR, betas=(0.5, 0.999))

    def lr_lambda(ep):
        half = CYC_EPOCHS // 2
        return 1.0 if ep < half else 1.0 - (ep - half) / half
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    mse = nn.MSELoss()   # LSGAN
    l1  = nn.L1Loss()

    dl_sim  = DataLoader(
        TensorDataset(torch.tensor(X_sim[:, None, :])),
        batch_size=CYC_BATCH, shuffle=True, drop_last=True)
    dl_real = DataLoader(
        TensorDataset(torch.tensor(X_real[:, None, :])),
        batch_size=CYC_BATCH, shuffle=True, drop_last=True)

    print(f'Training CycleGAN on {DEVICE} for {CYC_EPOCHS} epochs ...')
    for ep in range(1, CYC_EPOCHS + 1):
        g_sum = d_sum = n = 0
        for (s,), (r,) in zip(dl_sim, dl_real):
            s, r = s.to(DEVICE), r.to(DEVICE)
            B = s.size(0)

            # ── Generators ────────────────────────────────────────────
            opt_G.zero_grad()
            fake_r = G_S2R(s); fake_s = G_R2S(r)
            rec_s  = G_R2S(fake_r); rec_r  = G_S2R(fake_s)
            id_r   = G_S2R(r);  id_s   = G_R2S(s)

            loss_G = (
                mse(D_R(fake_r), torch.ones_like(D_R(fake_r))) +
                mse(D_S(fake_s), torch.ones_like(D_S(fake_s))) +
                LAMBDA_CYC * (l1(rec_s, s) + l1(rec_r, r)) +
                LAMBDA_ID  * (l1(id_r,  r) + l1(id_s,  s))
            )
            loss_G.backward(); opt_G.step()

            # ── Discriminators ────────────────────────────────────────
            opt_D.zero_grad()
            loss_D = 0.5 * (
                mse(D_R(r),              torch.ones_like(D_R(r))) +
                mse(D_R(fake_r.detach()), torch.zeros_like(D_R(fake_r))) +
                mse(D_S(s),              torch.ones_like(D_S(s))) +
                mse(D_S(fake_s.detach()), torch.zeros_like(D_S(fake_s)))
            )
            loss_D.backward(); opt_D.step()

            g_sum += loss_G.item(); d_sum += loss_D.item(); n += 1

        sched_G.step(); sched_D.step()
        if ep % 20 == 0 or ep == 1:
            print(f'  Ep {ep:3d}/{CYC_EPOCHS}  G={g_sum/n:.4f}  D={d_sum/n:.4f}')

    return G_S2R


# ═══════════════════════════════════════════════════════════════════════════
# STFT + CA-CNN (same architecture as method_d_cond)
# ═══════════════════════════════════════════════════════════════════════════

def make_stft(trace):
    f, _, Zxx = scipy_stft(trace, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                           nfft=NFFT, window='hann', boundary=None, padded=False)
    mag = np.abs(Zxx)[f <= FREQ_MAX, :]
    return (mag / (mag.max() + 1e-12)).astype(np.float32)


class GPRDataset(Dataset):
    def __init__(self, X, y, soil_arr, pipe_arr):
        self.imgs = [make_stft(x) for x in X]
        self.y    = torch.tensor(y,         dtype=torch.float32)
        self.soil = torch.tensor(soil_arr,  dtype=torch.long)
        self.pipe = torch.tensor(pipe_arr,  dtype=torch.long)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return (torch.tensor(self.imgs[i]).unsqueeze(0),
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


def run_cv(X_real, y_real, cond_real, X_trans, y_trans):
    """
    5-fold CV (real only for eval).
    Sim samples: condition unknown → assign sand/4in (soil=0, pipe=1), majority class.
    """
    n_sim    = len(X_trans)
    sim_soil = np.zeros(n_sim, dtype=np.int64)
    sim_pipe = np.ones(n_sim,  dtype=np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y = []; all_yhat = []; fold_r2 = []

    print(f'\n{"Fold":>5}  {"S":>7}  {"T":>7}  {"M":>7}  {"B":>7}  {"Avg":>7}')
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_real, cond_real), 1):
        # Build train set: real fold + all translated sim
        X_tr  = np.vstack([X_real[tr_idx], X_trans])
        y_tr  = np.vstack([y_real[tr_idx], y_trans])
        s_tr  = np.concatenate([[SOIL_MAP[c] for c in cond_real[tr_idx]], sim_soil])
        p_tr  = np.concatenate([[PIPE_MAP[c] for c in cond_real[tr_idx]], sim_pipe])

        tr_dl = DataLoader(GPRDataset(X_tr, y_tr, s_tr, p_tr),
                           batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(
            GPRDataset(X_real[va_idx], y_real[va_idx],
                       [SOIL_MAP[c] for c in cond_real[va_idx]],
                       [PIPE_MAP[c] for c in cond_real[va_idx]]),
            batch_size=BATCH)

        model = CNN2DCondV2().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=30, min_lr=1e-5)
        crit  = nn.MSELoss()
        best_val = float('inf'); best_wts = None; no_impr = 0

        for ep in range(1, EPOCHS + 1):
            model.train()
            for imgs, soil, pipe, ys in tr_dl:
                imgs, soil, pipe, ys = (imgs.to(DEVICE), soil.to(DEVICE),
                                        pipe.to(DEVICE), ys.to(DEVICE))
                opt.zero_grad()
                crit(model(imgs, soil, pipe), ys).backward()
                opt.step()

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for imgs, soil, pipe, ys in va_dl:
                    imgs, soil, pipe, ys = (imgs.to(DEVICE), soil.to(DEVICE),
                                            pipe.to(DEVICE), ys.to(DEVICE))
                    val_loss += crit(model(imgs, soil, pipe), ys).item() * len(imgs)
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
            for imgs, soil, pipe, _ in va_dl:
                preds.append(model(imgs.to(DEVICE), soil.to(DEVICE),
                                   pipe.to(DEVICE)).cpu().numpy())
        preds = np.vstack(preds)
        r2s   = [r2_score(y_real[va_idx][:, i], preds[:, i]) for i in range(4)]
        fold_r2.append(np.mean(r2s))
        all_y.append(y_real[va_idx]); all_yhat.append(preds)
        print(f'{fold:>5}  {r2s[0]:>7.3f}  {r2s[1]:>7.3f}  '
              f'{r2s[2]:>7.3f}  {r2s[3]:>7.3f}  {np.mean(r2s):>7.3f}')

    all_y    = np.vstack(all_y); all_yhat = np.vstack(all_yhat)
    r2_final = [r2_score(all_y[:, i], all_yhat[:, i]) for i in range(4)]
    return r2_final, np.std(fold_r2)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    X_real = np.load(DATA / 'X_norm.npy')          # (135, 256)
    y_real = np.load(DATA / 'y_moisture.npy')       # (135, 4)
    cond   = np.load(DATA / 'cond_labels.npy')      # (135,)
    X_sim  = np.load(SIM  / 'X_sim.npy')            # (375, 256)
    y_sim  = np.load(SIM  / 'y_sim.npy')            # (375, 4)

    print(f'Device: {DEVICE}')
    print(f'Real: {X_real.shape}  Sim: {X_sim.shape}')

    # Step 1: Train CycleGAN
    G_S2R = train_cyclegan(X_sim, X_real)

    # Step 2: Translate all sim samples → real style
    G_S2R.eval()
    with torch.no_grad():
        X_trans = G_S2R(
            torch.tensor(X_sim[:, None, :]).to(DEVICE)
        ).cpu().numpy()[:, 0, :]   # (375, 256)
    print(f'\nTranslated range: [{X_trans.min():.3f}, {X_trans.max():.3f}]')
    np.save(SIM / 'X_sim_translated.npy', X_trans)
    print('Saved: data/sim/X_sim_translated.npy')

    # Step 3: 5-fold CV with translated sim as augmentation
    print('\n=== 5-fold CV: CA-CNN + CycleGAN-translated sim ===')
    r2_final, std_r2 = run_cv(X_real, y_real, cond, X_trans, y_sim)
    avg_r2 = np.mean(r2_final)

    print(f'\n  S={r2_final[0]:.3f}  T={r2_final[1]:.3f}  '
          f'M={r2_final[2]:.3f}  B={r2_final[3]:.3f}  '
          f'Avg={avg_r2:.3f} ± {std_r2:.3f}')

    print('\n=== 对比 ===')
    print(f'CA-CNN only (no aug):           R² = 0.816')
    print(f'+ cWGAN-GP aug:                 R² = 0.825  ← current best')
    print(f'+ Sim pre-train (直接迁移):      R² = 0.749')
    print(f'+ CycleGAN Sim2Real (本脚本):   R² = {avg_r2:.3f}')
