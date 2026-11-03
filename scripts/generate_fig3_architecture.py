"""
Generate Fig 3: CA-CNN architecture diagram for the paper.
Saves to paper/figures/fig3_architecture.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

# ── Color palette ──────────────────────────────────────────────────────────
C_INPUT  = '#c8e6c9'   # green-tinted: input
C_CONV   = '#bbdefb'   # blue: conv blocks
C_POOL   = '#fff9c4'   # yellow: pool
C_COND   = '#ffe0b2'   # orange: condition
C_FC     = '#e1bee7'   # purple: FC
C_OUT    = '#ffcdd2'   # red: output
C_ARROW  = '#455a64'   # dark slate

def box(ax, x, y, w, h, color, label, sublabel='', fontsize=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.04',
                          facecolor=color, edgecolor='#546e7a', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#1a1a1a')
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha='center', va='center', fontsize=7.5, color='#444444')

def arrow(ax, x1, x2, y=2.5, color=C_ARROW):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def vline(ax, x, y_bottom, y_top, color=C_ARROW):
    ax.plot([x, x], [y_bottom, y_top], color=color, lw=1.4, ls='--', alpha=0.7)

Y = 2.5   # main row y

# ── 1. Input ───────────────────────────────────────────────────────────────
box(ax, 0.75, Y, 1.1, 1.0, C_INPUT, 'STFT', '153 × 57')
arrow(ax, 1.3, 1.9)

# ── 2. Conv blocks ─────────────────────────────────────────────────────────
for i, (cx, label, sub) in enumerate([
    (2.4,  'Conv-BN\nReLU-Pool', '1→16'),
    (3.85, 'Conv-BN\nReLU-Pool', '16→32'),
    (5.3,  'Conv-BN\nReLU-Pool', '32→64'),
]):
    box(ax, cx, Y, 1.1, 1.0, C_CONV, label, sub, fontsize=8.5)
    if i < 2:
        arrow(ax, cx + 0.55, cx + 1.3 - 0.55)

arrow(ax, 5.85, 6.5)
box(ax, 6.9, Y, 0.8, 0.8, C_POOL, 'AvgPool', '512-d', fontsize=8.5)
arrow(ax, 7.3, 7.9)

# ── 3. Condition one-hot vectors ───────────────────────────────────────────
box(ax, 8.4,  3.9, 1.0, 0.65, C_COND, 'Soil type', 'one-hot (2)', fontsize=8)
box(ax, 8.4,  3.1, 1.0, 0.65, C_COND, 'Pipe dia.', 'one-hot (2)', fontsize=8)

# dashed lines from condition boxes to concat point
ax.annotate('', xy=(9.1, Y + 0.18), xytext=(8.9, 3.6),
            arrowprops=dict(arrowstyle='->', color='#ff7043', lw=1.3, linestyle='dashed'))
ax.annotate('', xy=(9.1, Y - 0.1), xytext=(8.9, 3.14),
            arrowprops=dict(arrowstyle='->', color='#ff7043', lw=1.3, linestyle='dashed'))

# concat symbol
box(ax, 9.4, Y, 0.55, 0.9, '#eceff1', 'cat', '516-d', fontsize=8)
ax.text(9.4, Y + 0.0, '⊕', ha='center', va='center', fontsize=16, color='#37474f')
arrow(ax, 9.68, 10.2)

# ── 4. FC layers ───────────────────────────────────────────────────────────
box(ax, 10.65, Y, 0.85, 0.9, C_FC, 'FC-128', 'ReLU\nDrop 0.3', fontsize=8.5)
arrow(ax, 11.08, 11.65)
box(ax, 12.1,  Y, 0.8,  0.9, C_FC, 'FC-4', '', fontsize=8.5)
arrow(ax, 12.5, 13.1)

# ── 5. Output ──────────────────────────────────────────────────────────────
box(ax, 13.5, Y, 0.85, 1.1, C_OUT, 'Output', 'θ_S, θ_T\nθ_M, θ_B', fontsize=8.5)

# ── Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor='#546e7a', label='Input'),
    mpatches.Patch(facecolor=C_CONV,  edgecolor='#546e7a', label='Conv block'),
    mpatches.Patch(facecolor=C_COND,  edgecolor='#546e7a', label='Condition'),
    mpatches.Patch(facecolor=C_FC,    edgecolor='#546e7a', label='FC layer'),
    mpatches.Patch(facecolor=C_OUT,   edgecolor='#546e7a', label='Output'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=5,
          bbox_to_anchor=(0.5, -0.08), framealpha=0.9, fontsize=8.5)

# ── Bracket label for conv blocks ──────────────────────────────────────────
ax.annotate('', xy=(5.9, 1.65), xytext=(1.85, 1.65),
            arrowprops=dict(arrowstyle='<->', color='#607d8b', lw=1.2))
ax.text(3.87, 1.42, '3 convolutional blocks', ha='center', fontsize=8, color='#546e7a', style='italic')

ax.set_title('Fig. 3  CA-CNN Architecture', fontsize=11, fontweight='bold', pad=8, color='#1a1a1a')

plt.tight_layout()
plt.savefig(OUT / 'fig3_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT / "fig3_architecture.png"}')
