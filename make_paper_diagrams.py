"""Two additional paper diagrams (beyond fig1-4), shared style from figstyle.py:

fig0_mechanism : two panels. (a) the non-monotone required slope kappa(t)=s/gamma^2 with
                 a finite ceiling kbar -> FINITE deficit band (Lemma 3) + the freeze-in;
                 (b) the closed-form floor g(rho) for both classes (Thm 2).
fig5_pipeline  : schematic of the self-consuming loop and where the anchor sits.

All quantitative curves are computed from the real formulas (no cartoons for math).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as st

st.use_style()

# ---------------------------------------------------------------- fig0 (a)+(b)
SIG = 0.05
t = np.geomspace(1e-6, 8, 4000)
s = np.sqrt(1 - np.exp(-t))
g2 = np.exp(-t) * SIG ** 2 + s ** 2
kap = s / g2
KBAR = 3.9

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.55))

# --- panel (a): the required slope is non-monotone, so the deficit band is finite.
# Direct labels only (no legend box), so nothing can collide.
ax.plot(t, kap, color=st.BLUE, lw=1.8)
ax.axhline(KBAR, color=st.RED, lw=1.3, ls="--")
bandmask = kap > KBAR
tlo, thi = t[bandmask].min(), t[bandmask].max()
ax.axvspan(tlo, thi, color=st.RED, alpha=0.08, lw=0)
ax.text(np.sqrt(tlo * thi), 6.5, "deficit band:\nvariance freezes in",
        ha="center", va="center", fontsize=8, color=st.RED)
ax.annotate("peak $1/(2\\sigma)$", xy=(SIG ** 2, 1 / (2 * SIG)),
            xytext=(1.2e-6, 9.6), fontsize=8, color=st.GRAY,
            arrowprops=dict(arrowstyle="->", lw=0.9, color=st.GRAY))
ax.text(1.0, 0.38, "required slope $\\kappa(t)$", fontsize=8,
        color=st.BLUE, ha="center", va="bottom")
ax.text(6.5, KBAR + 0.2, "network cap $\\bar\\kappa$", fontsize=8,
        color=st.RED, ha="right", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("diffusion time $t$ (log scale)")
ax.set_ylabel("slope $\\kappa$")
ax.set_ylim(0, 11.4)
ax.set_title("(a)", loc="left")

# --- panel (b): the closed-form floor as a function of the slope cap
rho = np.linspace(0.02, 0.999, 500)
q = np.sqrt(1 - rho ** 2)
xlo = (1 - q) / rho; xhi = (1 + q) / rho
ulo = xlo ** 2; uhi = xhi ** 2
E = np.exp(-4 * q)
Wex = (1 + uhi) * E + ((3 - 2 * q) - (3 + 2 * q) * E) / (2 * rho ** 2)
g_no = 1 + (Wex - (1 + ulo)) / (1 + ulo) ** 2
g_un = 1 / (2 * rho ** 2)

ax2.plot(rho, g_no, color=st.BLUE, lw=1.8, label="no-overshoot: $g(\\rho)$")
ax2.plot(rho, g_un, color=st.AMBER, lw=1.6, ls="--", label="unconditional: $1/(2\\rho^2)$")
fs = [17.005, 14.121, 13.571]  # rho=0.2 at sigma=0.10, 0.05, 0.02
ax2.plot([0.20] * 3, fs, "o", color=st.GREEN, ms=4.5,
         label="finite-$\\sigma$ ODE ($\\sigma$=0.10, 0.05, 0.02)")
ax2.axhline(1, color=st.GRAY, ls=":", lw=1.0)
ax2.text(0.03, 1.15, "true width $\\sigma^2$", color=st.GRAY, ha="left", fontsize=8)
ax2.set_yscale("log")
ax2.set_xlabel("slope cap / peak, $\\rho=\\bar\\kappa/\\kappa_{\\max}$")
ax2.set_ylabel("floor width / $\\sigma^2$")
ax2.set_title("(b)", loc="left")
ax2.legend(loc="upper right")
ax2.set_xlim(0, 1.02); ax2.set_ylim(0.4, 300)
ax2.grid(axis="y", lw=0.5, alpha=0.18)

fig.tight_layout(w_pad=2.0)
st.save(fig, "fig0_mechanism")
print("fig0 OK")

# ---------------------------------------------------------------- fig5 pipeline
fig, ax = plt.subplots(figsize=(6.2, 2.35))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2, color="black", lw=1.1, con="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", lw=lw,
                                 color=color, connectionstyle=con,
                                 mutation_scale=11, shrinkA=2, shrinkB=2))

ROW_Y, ROW_H = 1.45, 0.95
box(0.15, ROW_Y, 1.55, ROW_H, "fresh real\ndata ($\\lambda$)", st.LIGHT["blue"], st.BLUE)
box(2.30, ROW_Y, 1.45, ROW_H, "pool$_g$", st.LIGHT["gray"], st.GRAY)
box(4.35, ROW_Y, 1.60, ROW_H, "train score\n$\\hat\\varepsilon_g$ (DSM)", st.LIGHT["gray"], st.GRAY)
box(6.55, ROW_Y, 1.60, ROW_H, "sample\n(reverse SDE)", st.LIGHT["gray"], st.GRAY)
box(8.75, ROW_Y, 1.10, ROW_H, "gen$_g$", st.LIGHT["red"], st.RED)
mid = ROW_Y + ROW_H / 2
arrow(1.75, mid, 2.26, mid)
arrow(3.80, mid, 4.31, mid)
arrow(6.00, mid, 6.51, mid)
arrow(8.20, mid, 8.71, mid)

# recursion arrow (bulges below the row)
arrow(9.30, ROW_Y - 0.06, 3.05, ROW_Y - 0.06, color=st.RED, con="arc3,rad=-0.25")
ax.text(6.15, 0.28, "self-consuming recursion: gen$_g$ becomes $(1-\\lambda)$ of pool$_{g+1}$",
        color=st.RED, fontsize=8, ha="center")

# anchor row (top), with clear margin from the canvas edge
TOP_Y, TOP_H = 3.05, 0.80
box(0.15, TOP_Y, 2.30, TOP_H, "stale real reference\n($m$=2000, from $g$=0)",
    st.LIGHT["green"], st.GREEN, fs=8)
box(3.05, TOP_Y, 3.55, TOP_H, "anchor: bidirectional per-axis\nlocal moment match (Thm 3)",
    st.LIGHT["green"], st.GREEN, fs=8)
arrow(2.47, TOP_Y + TOP_H / 2, 3.01, TOP_Y + TOP_H / 2, color=st.GREEN)
arrow(3.85, TOP_Y - 0.04, 3.10, ROW_Y + ROW_H + 0.04, color=st.GREEN, con="arc3,rad=0.18")
arrow(6.20, TOP_Y - 0.04, 8.85, ROW_Y + ROW_H + 0.04, color=st.GREEN, con="arc3,rad=-0.22")
ax.text(8.25, TOP_Y + 0.42, "matches pool + output\nnormal variance\nto the reference",
        fontsize=8, color=st.GREEN, ha="center", va="center")

st.save(fig, "fig5_pipeline")
print("fig5 OK")
