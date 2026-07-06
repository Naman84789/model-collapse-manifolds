"""CIFAR-10 pixel-recursion figure for the paper (fig6_cifar.png).
Two panels, both honest:
  (a) off-manifold distance vs generation, unanchored vs anchored, 3 seeds (mean +/- s.d.),
      true-data baseline line. Shows the stable-plateau vs wander dynamics at D=3072.
  (b) three nearest-neighbor metrics vs the real-vs-real baseline (=1.0): precision,
      COVERAGE (the mode-collapse detector), diversity. Coverage=1.0 for the anchor
      rules out mode collapse; diversity 0.75x is the bounded contraction cost.
Data are the measured trajectories (cifar_run.log) and diversity checks
(cifar_diversity_check.py, 3 seeds), hardcoded so the figure regenerates with no state files.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(BASE, "paper", "figs")
os.makedirs(OUTD, exist_ok=True)
TRUE = 20.1490

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
C_BAD = "#d62728"; C_GOOD = "#2ca02c"; C_TRUE = "#444444"

# ---- panel (a) data: off-manifold distance, g0..g7 ----
UNF = np.array([
 [20.3693,20.7590,20.9482,24.1950,22.4064,21.5588,23.6090,21.6922],
 [20.5738,22.4452,22.3693,23.7238,22.8924,25.3032,25.2588,23.8071],
 [22.0095,22.8561,22.3604,22.4769,23.9381,22.0913,21.5545,21.6605]])
FIX = np.array([
 [18.4863,17.7967,16.1853,16.9317,16.9281,16.2968,16.1844,16.3205],
 [15.9864,15.9193,14.8740,15.6840,15.5338,16.3904,16.0388,15.5606],
 [16.7197,15.9353,15.8137,17.4464,16.6250,16.9882,16.4627,16.3018]])
g = np.arange(8)

# ---- panel (b) data: metrics vs real-vs-real baseline (mean over 3 seeds) ----
# rows: precision, coverage, diversity ; cols: unfixed, fixed
metrics = ["precision\n(gen to ref)", "COVERAGE\n(real to gen)", "diversity\n(gen intra)"]
unf_m = [1.11, 1.16, 1.20]
fix_m = [0.80, 1.00, 0.75]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

def band(ax, x, T, color, label, ls="-"):
    m = T.mean(0); s = T.std(0)
    ax.plot(x, m, color=color, lw=2.3, ls=ls, label=label, zorder=3)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, zorder=2)

# panel a
band(ax1, g, UNF, C_BAD, "unanchored  (degrades)")
band(ax1, g, FIX, C_GOOD, "+ local anchor  (ours)")
ax1.axhline(TRUE, color=C_TRUE, ls=":", lw=1.6)
ax1.text(0, TRUE + 0.15, "true-data baseline %.1f" % TRUE, ha="left", va="bottom",
         color=C_TRUE, fontsize=10)
ax1.set_xlabel("generation of self-consuming training")
ax1.set_ylabel("off-manifold distance to real CIFAR-10")
ax1.set_title("(a) CIFAR-10 pixels (D=3072): anchor holds a stable plateau")
ax1.legend(loc="upper right", framealpha=0.95)
ax1.set_xticks(g); ax1.set_ylim(14, 26)

# panel b
x = np.arange(3); w = 0.36
ax2.bar(x - w/2, unf_m, w, color=C_BAD, alpha=0.85, label="unanchored")
ax2.bar(x + w/2, fix_m, w, color=C_GOOD, alpha=0.85, label="+ local anchor")
ax2.axhline(1.0, color=C_TRUE, ls=":", lw=1.6)
ax2.text(2.55, 1.02, "real-data\nbaseline", color=C_TRUE, fontsize=9, ha="right", va="bottom")
for xi, (u, f) in enumerate(zip(unf_m, fix_m)):
    ax2.text(xi - w/2, u + 0.02, "%.2f" % u, ha="center", va="bottom", fontsize=9, color=C_BAD)
    ax2.text(xi + w/2, f + 0.02, "%.2f" % f, ha="center", va="bottom", fontsize=9, color=C_GOOD)
ax2.annotate("coverage = 1.00:\nno mode collapse", xy=(1 + w/2, 1.00), xytext=(1.15, 0.55),
             fontsize=9.5, color=C_GOOD,
             arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.3))
ax2.set_xticks(x); ax2.set_xticklabels(metrics)
ax2.set_ylabel("nearest-neighbor metric  /  real baseline")
ax2.set_title("(b) The anchor keeps full coverage (rules out collapse)")
ax2.legend(loc="upper left", framealpha=0.95)
ax2.set_ylim(0, 1.4)

fig.tight_layout()
out = os.path.join(OUTD, "fig6_cifar.png")
fig.savefig(out); plt.close(fig)
print("fig6_cifar.png written to", out)
print("  unanchored g1-g7 mean = %.2f, anchored = %.2f, gap = %.2f"
      % (UNF[:, 1:].mean(), FIX[:, 1:].mean(), UNF[:, 1:].mean() - FIX[:, 1:].mean()))
