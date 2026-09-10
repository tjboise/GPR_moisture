"""
Visualization of the merged dataset (original 112 + additional 23 = 135 samples).
Generates fig_merged_overview.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

X      = np.load('../data/X_norm.npy')       # (135, 256)
y      = np.load('../data/y_moisture.npy')   # (135, 4)
cond   = np.load('../data/cond_labels.npy')  # (135,)
idx_tr = np.load('../data/idx_train.npy')
idx_te = np.load('../data/idx_test.npy')
time   = np.linspace(0, 255 * 0.099609, 256)

LAYERS   = ['S (0 cm)', 'T (8 cm)', 'M (22 cm)', 'B (35 cm)']
COND_NAMES = ['2-in Sand\n(n=47)', '4-in Sand\n(n=48)', '4-in Clay\n(n=40)']
COND_COLS  = ['#2196F3', '#FF9800', '#4CAF50']
COND_LITE  = ['#BBDEFB', '#FFE0B2', '#C8E6C9']

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Merged Dataset Overview — 135 A-scans  (original 112 + additional 23)',
             fontsize=14, fontweight='bold', y=0.98)

gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
              left=0.06, right=0.97, top=0.93, bottom=0.07)

# ── Row 0: aligned A-scans per condition ──────────────────────────────────
for ci in range(3):
    ax = fig.add_subplot(gs[0, ci])
    idx = np.where(cond == ci)[0]
    for i in idx:
        ax.plot(time, X[i], color=COND_COLS[ci], alpha=0.25, lw=0.6)
    ax.plot(time, X[idx].mean(axis=0), color='black', lw=1.5, label='mean')
    ax.set_title(COND_NAMES[ci], fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (ns)', fontsize=8)
    ax.set_ylabel('Norm. amplitude', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7)

# Right of row 0: train/test split bar
ax = fig.add_subplot(gs[0, 3])
for ci in range(3):
    n_tr = np.sum(cond[idx_tr] == ci)
    n_te = np.sum(cond[idx_te] == ci)
    ax.bar(ci - 0.2, n_tr, 0.38, color=COND_COLS[ci],   label='Train' if ci == 0 else '')
    ax.bar(ci + 0.2, n_te, 0.38, color=COND_LITE[ci], edgecolor=COND_COLS[ci],
           lw=1.2, label='Test' if ci == 0 else '')
    ax.text(ci - 0.2, n_tr + 0.4, str(n_tr), ha='center', fontsize=8, fontweight='bold')
    ax.text(ci + 0.2, n_te + 0.4, str(n_te), ha='center', fontsize=8, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['2in Sand', '4in Sand', '4in Clay'], fontsize=8)
ax.set_ylabel('Count', fontsize=8); ax.set_title('Train / Test split', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3); ax.tick_params(labelsize=7)

# ── Rows 1-2: moisture distribution per layer × condition ─────────────────
for li, lname in enumerate(LAYERS):
    row = 1 + li // 2
    col = li % 2 * 2
    # Violin / strip plot
    ax = fig.add_subplot(gs[row, col:col+2])
    parts = ax.violinplot(
        [y[cond == ci, li] for ci in range(3)],
        positions=[0, 1, 2], widths=0.55, showmedians=True,
        showextrema=True)
    for ci, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COND_COLS[ci]); pc.set_alpha(0.55)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_lw(1.5)
    for ci in range(3):
        vals = y[cond == ci, li]
        ax.scatter(np.full(len(vals), ci) + np.random.uniform(-0.12, 0.12, len(vals)),
                   vals, color=COND_COLS[ci], s=18, alpha=0.7, edgecolors='white', lw=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['2in Sand', '4in Sand', '4in Clay'], fontsize=8)
    ax.set_ylabel('Moisture (%)', fontsize=8)
    ax.set_title(f'Moisture — {lname}', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2); ax.tick_params(labelsize=7)

plt.savefig('../results/fig_merged_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig_merged_overview.png')

# ── Print summary table ────────────────────────────────────────────────────
print()
print(f'{"":>18}  {"2in Sand":>10} {"4in Sand":>10} {"4in Clay":>10} {"All":>8}')
print('-' * 62)
print(f'{"Samples":>18}  {(cond==0).sum():>10} {(cond==1).sum():>10} {(cond==2).sum():>10} {len(cond):>8}')
print(f'{"  Train":>18}  {(cond[idx_tr]==0).sum():>10} {(cond[idx_tr]==1).sum():>10} {(cond[idx_tr]==2).sum():>10} {len(idx_tr):>8}')
print(f'{"  Test":>18}  {(cond[idx_te]==0).sum():>10} {(cond[idx_te]==1).sum():>10} {(cond[idx_te]==2).sum():>10} {len(idx_te):>8}')
print()
for li, lname in enumerate(LAYERS):
    row_parts = []
    for ci in range(3):
        v = y[cond == ci, li]
        row_parts.append(f'{v.mean():4.1f}±{v.std():3.1f}')
    v_all = y[:, li]
    print(f'{lname:>18}  {"  ".join(f"{p:>10}" for p in row_parts)}  {v_all.mean():4.1f}±{v_all.std():3.1f}')
