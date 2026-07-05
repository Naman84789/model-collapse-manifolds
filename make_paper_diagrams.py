"""Two additional paper diagrams (beyond fig1-4):

fig0_mechanism : two panels. (a) the non-monotone required slope kappa(t)=s/gamma^2 with
                 a finite ceiling kbar -> FINITE deficit band (Lemma 3.0) + the freeze-in;
                 (b) the closed-form floor g(rho) for both classes (Lemma 3.3).
fig5_pipeline  : schematic of the self-consuming loop and where the anchor sits.

All quantitative curves are computed from the real formulas (no cartoons for math).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\naman\Downloads\metric-audit\paper\figs"
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 13,
    "axes.titlesize": 14, "axes.labelsize": 13.5, "legend.fontsize": 12,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
BLUE, ORANGE, RED, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#555555"

# ---------------------------------------------------------------- fig0 (a)+(b)
SIG = 0.05
t = np.geomspace(1e-6, 8, 4000)
s = np.sqrt(1 - np.exp(-t))
g2 = np.exp(-t) * SIG ** 2 + s ** 2
kap = s / g2
KBAR = 3.9

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

# --- panel (a): the required slope is non-monotone, so the deficit band is finite
ax.plot(t, kap, color=BLUE, lw=2.6, label=r"required slope $\kappa(t)$")
ax.axhline(KBAR, color=RED, lw=2.0, ls="--", label=r"network cap $\bar\kappa$")
band = kap > KBAR
tlo, thi = t[band].min(), t[band].max()
ax.axvspan(tlo, thi, color=RED, alpha=0.12)
ax.annotate("deficit band:\nvariance freezes in",
            xy=(np.sqrt(tlo * thi), 6.2), ha="center", va="center",
            fontsize=12.5, color=RED, fontweight="bold")
ax.annotate(r"peak $=1/2\sigma$",
            xy=(SIG ** 2, 1 / (2 * SIG)), xytext=(7e-5, 9.2), fontsize=12, color=GRAY,
            arrowprops=dict(arrowstyle="->", lw=1.3, color=GRAY))
ax.set_xscale("log"); ax.set_xlabel(r"diffusion time $t$  (log scale)")
ax.set_ylabel(r"slope $\kappa$"); ax.set_ylim(0, 11)
ax.set_title("(a) the required slope is non-monotone")
ax.legend(loc="upper right", framealpha=0.95)

# --- panel (b): the closed-form floor as a function of the slope cap
rho = np.linspace(0.02, 0.999, 500)
q = np.sqrt(1 - rho ** 2)
xlo = (1 - q) / rho; xhi = (1 + q) / rho
ulo = xlo ** 2; uhi = xhi ** 2
E = np.exp(-4 * q)
Wex = (1 + uhi) * E + ((3 - 2 * q) - (3 + 2 * q) * E) / (2 * rho ** 2)
g_no = 1 + (Wex - (1 + ulo)) / (1 + ulo) ** 2
g_un = 1 / (2 * rho ** 2)

ax2.plot(rho, g_no, color=BLUE, lw=2.6, label=r"no-overshoot: $g(\rho)$")
ax2.plot(rho, g_un, color=ORANGE, lw=2.6, ls="--", label=r"unconditional: $1/(2\rho^2)$")
ax2.axhline(1, color=GRAY, ls=":", lw=1.6)
ax2.text(0.98, 1.18, r"true width $\sigma^2$", color=GRAY, ha="right", fontsize=11.5)
fs = [17.005, 14.121, 13.571]
for i, v in enumerate(fs):
    ax2.plot([0.20], [v], "o", color=GREEN, ms=8,
             label=("finite-$\\sigma$ (numerical)" if i == 0 else "_nolegend_"))
ax2.set_yscale("log")
ax2.set_xlabel(r"slope cap / peak,  $\rho=\bar\kappa/\kappa_{\max}$")
ax2.set_ylabel(r"floor width / $\sigma^2$")
ax2.set_title("(b) the floor is closed-form in the cap")
ax2.legend(loc="upper right", framealpha=0.95)
ax2.set_xlim(0, 1.02); ax2.set_ylim(0.4, 300)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig0_mechanism.png")); plt.close(fig)
print("fig0 OK")

# ---------------------------------------------------------------- fig5 pipeline
fig, ax = plt.subplots(figsize=(9.0, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=10, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2, color="black", style="-|>", lw=1.8, con="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=lw,
                                 color=color, connectionstyle=con, mutation_scale=16))

box(0.15, 1.45, 1.55, 0.9, "fresh real\ndata  ($\\lambda$)", "#eaf3fb", BLUE)
box(2.35, 1.45, 1.45, 0.9, "pool$_g$", "#f5f5f5", GRAY)
box(4.45, 1.45, 1.55, 0.9, "train score\n$\\hat\\varepsilon_g$ (DSM)", "#f5f5f5", GRAY)
box(6.65, 1.45, 1.55, 0.9, "sample\n(reverse SDE)", "#f5f5f5", GRAY)
box(8.75, 1.45, 1.1, 0.9, "gen$_g$", "#fdeaea", RED)
arrow(1.7, 1.9, 2.32, 1.9)
arrow(3.8, 1.9, 4.42, 1.9)
arrow(6.0, 1.9, 6.62, 1.9)
arrow(8.2, 1.9, 8.72, 1.9)
# recursion arrow (bulges BELOW the row)
arrow(9.25, 1.42, 3.1, 1.42, color=RED, con="arc3,rad=-0.3")
ax.text(6.15, 0.35, "self-consuming recursion:  gen$_g$ becomes $(1-\\lambda)$ of pool$_{g+1}$",
        color=RED, fontsize=9.5, ha="center")

# anchor row (top)
box(0.15, 2.75, 2.35, 0.62, "stale real reference\n($m$=2000, from $g$=0)", "#eafbea", GREEN, fs=8.8)
box(3.15, 2.75, 4.0, 0.62,
    "ANCHOR: bidirectional per-axis\nlocal moment match (Thm 3)", "#eafbea", GREEN, fs=9)
arrow(2.52, 3.06, 3.12, 3.06, color=GREEN)
arrow(4.0, 2.72, 3.1, 2.4, color=GREEN, con="arc3,rad=0.2")
arrow(6.4, 2.72, 8.6, 2.4, color=GREEN, con="arc3,rad=-0.2")
ax.text(8.0, 2.95, "matches pool + output normal\nvariance to the reference",
        fontsize=8.3, color=GREEN, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_pipeline.png")); plt.close(fig)
print("fig5 OK")

# copy the four result figures into paper/figs
import shutil
SRC = r"C:\Users\naman\Downloads\metric-audit\figures"
for f in os.listdir(SRC):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
print("figs copied:", sorted(os.listdir(OUT)))
