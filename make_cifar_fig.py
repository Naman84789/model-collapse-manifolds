"""CIFAR-10 pixel-recursion figure for the paper (fig6_cifar), shared style from
figstyle.py. Two panels, both honest:
  (a) off-manifold distance vs generation, unanchored vs anchored, 3 seeds (mean +/- s.d.),
      true-data baseline line. Shows the stable-plateau vs wander dynamics at D=3072.
  (b) three nearest-neighbor metrics vs the real-vs-real baseline (=1.0): precision,
      coverage (the mode-collapse detector), diversity. Coverage=1.0 for the anchor
      rules out mode collapse; diversity 0.75x is the bounded contraction cost.
Data are the measured trajectories (cifar_run.log) and diversity checks
(cifar_diversity_check.py, 3 seeds), hardcoded so the figure regenerates with no state files.
"""
import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.use_style()
TRUE = 20.1490

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
metrics = ["precision\n(gen to ref)", "coverage\n(real to gen)", "diversity\n(gen intra)"]
unf_m = [1.11, 1.16, 1.20]
fix_m = [0.80, 1.00, 0.75]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

# panel a
st.band(ax1, g, UNF, st.RED, "unanchored (degrades)")
st.band(ax1, g, FIX, st.GREEN, "+ local anchor (ours)")
ax1.axhline(TRUE, color=st.GRAY, ls=":", lw=1.0)
ax1.text(7, TRUE - 0.25, "true-data baseline %.1f" % TRUE, ha="right", va="top",
         color=st.GRAY, fontsize=8)
ax1.set_xlabel("generation of self-consuming training")
ax1.set_ylabel("off-manifold distance\nto real CIFAR-10")
ax1.set_title("(a)", loc="left")
ax1.legend(loc="upper right")
ax1.set_xticks(g); ax1.set_ylim(14, 27)
ax1.grid(axis="y", lw=0.5, alpha=0.18)

# panel b
x = np.arange(3); w = 0.36
ax2.bar(x - w/2, unf_m, w, color=st.RED, label="unanchored")
ax2.bar(x + w/2, fix_m, w, color=st.GREEN, label="+ local anchor")
ax2.axhline(1.0, color=st.GRAY, ls=":", lw=1.0)
for xi, (u, f) in enumerate(zip(unf_m, fix_m)):
    ax2.text(xi - w/2, u + 0.02, "%.2f" % u, ha="center", va="bottom", fontsize=7.5)
    ax2.text(xi + w/2, f + 0.02, "%.2f" % f, ha="center", va="bottom", fontsize=7.5)
ax2.text(1.0, 1.26, "coverage 1.00:\nno mode collapse", ha="center", va="bottom",
         fontsize=8, color=st.GREEN)
ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=8)
ax2.set_ylabel("NN metric / real baseline (dotted)")
ax2.set_title("(b)", loc="left")
ax2.legend(ncol=2, loc="upper center")
ax2.set_ylim(0, 1.62)

fig.tight_layout(w_pad=1.6)
st.save(fig, "fig6_cifar")
print("fig6 OK  unanchored g1-g7 mean = %.2f, anchored = %.2f, gap = %.2f"
      % (UNF[:, 1:].mean(), FIX[:, 1:].mean(), UNF[:, 1:].mean() - FIX[:, 1:].mean()))
