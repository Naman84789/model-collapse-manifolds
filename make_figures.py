"""Paper figures from the verified jsonl runs. Each figure is self-contained and guarded
so a missing input can't kill the others. Outputs 300-dpi PNGs to metric-audit/figures/.

Fig 1  head-to-head        : Cambridge annealed truncation vs our anchored fix (rvar/sigma^2)
Fig 2  two-horned trap      : deterministic->point, stochastic->fat, anchored->sigma^2
Fig 3  pixel MNIST          : unfixed drift vs fixed vs true-data baseline (off-manifold)
Fig 4  deterministic law    : Phi_det/sigma^2 = g(kbar), measured point + 0.141/kbar^2 asymptote
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = r"."
OUTD = os.path.join(D, "figures"); os.makedirs(OUTD, exist_ok=True)
SIG2 = 0.05 ** 2

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})
C_BAD = "#d62728"; C_STD = "#1f77b4"; C_JMP = "#2ca02c"
C_DET = "#ff7f0e"; C_TRUE = "#444444"; C_ANCH = "#1f77b4"

def load(name):
    rows = {}
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "key" in d: rows[d["key"]] = d
    return rows

def band(ax, x, trajs, color, label, lw=2.2, ls="-"):
    T = np.array(trajs); m = T.mean(0); s = T.std(0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label, zorder=3)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, zorder=2)
    return m

# ---------------------------------------------------------------- Fig 1
try:
    r = load("head_to_head.jsonl")
    A = [r[k]["traj"] for k in r if k.startswith("A_")]
    B = [r[k]["traj"] for k in r if k.startswith("B_")]
    Cc = [r[k]["traj"] for k in r if k.startswith("C_")]
    g = np.arange(len(A[0]))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    band(ax, g, np.array(A) / SIG2, C_BAD, "Cambridge: annealed truncation (no anchor)")
    band(ax, g, np.array(B) / SIG2, C_STD, "Ours: std sampler + anchor")
    band(ax, g, np.array(Cc) / SIG2, C_JMP, "Ours: jump sampler + anchor", ls="--")
    ax.axhline(1.0, color=C_TRUE, ls=":", lw=1.5, zorder=1)
    ax.text(len(g) - 1, 1.06, "true tube  $\\sigma^2$", ha="right", va="bottom",
            color=C_TRUE, fontsize=10)
    ax.set_yscale("log"); ax.set_ylim(0.7, 60)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("output variance  /  $\\sigma^2$")
    ax.set_title("Anchor holds at $\\sigma^2$; annealed truncation stays ~10× fat")
    ax.legend(loc="center left", framealpha=0.95)
    ax.set_xticks(g[::2])
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, "fig1_head_to_head.png"))
    plt.close(fig)
    print("Fig 1 OK  (Cambridge g19 = %.1f s2, ours = %.2f s2)"
          % (np.array(A).mean(0)[-1] / SIG2, np.array(B).mean(0)[-1] / SIG2))
except Exception as e:
    print("Fig 1 FAILED:", e)

# ---------------------------------------------------------------- Fig 2
try:
    r = load("inject_ablation.jsonl")
    NO = [r[k]["traj"] for k in r if k.startswith("noise_")]
    DE = [r[k]["traj"] for k in r if k.startswith("nonoise_")]
    rh = load("head_to_head.jsonl")
    B = [rh[k]["traj"] for k in rh if k.startswith("B_")]
    g = np.arange(len(NO[0]))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhspan(1.0, 60, color=C_BAD, alpha=0.05)
    ax.axhspan(1e-3, 1.0, color=C_DET, alpha=0.05)
    band(ax, g, np.array(NO) / SIG2, C_BAD, "stochastic sampler  →  stuck fat (~11$\\sigma^2$)")
    mB = np.array(B)[:, :len(g)]
    band(ax, g, mB / SIG2, C_ANCH, "+ anchor  →  $\\sigma^2$")
    # deterministic collapses to 0 -> represent on symlog floor with annotation
    mDE = np.array(DE).mean(0) / SIG2
    ax.plot(g, np.clip(mDE, 3e-3, None), color=C_DET, lw=2.2,
            label="deterministic sampler  →  point collapse (V→0)")
    ax.annotate("collapse to a single point", xy=(2, 4e-3), xytext=(6, 0.02),
                color=C_DET, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_DET, lw=1.4))
    ax.axhline(1.0, color=C_TRUE, ls=":", lw=1.5)
    ax.text(0.2, 1.15, "true tube  $\\sigma^2$", color=C_TRUE, fontsize=10)
    ax.text(len(g) - 1, 20, "TOO FAT", ha="right", color=C_BAD, fontsize=10, alpha=0.7)
    ax.text(len(g) - 1, 0.02, "TOO THIN", ha="right", color=C_DET, fontsize=10, alpha=0.7)
    ax.set_yscale("log"); ax.set_ylim(3e-3, 60)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("output variance  /  $\\sigma^2$")
    ax.set_title("The two-horned trap: neither unanchored sampler reaches $\\sigma^2$")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xticks(g[::2])
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, "fig2_two_horned_trap.png"))
    plt.close(fig)
    print("Fig 2 OK  (noise=%.1f s2, nonoise final=%.3f s2)"
          % (np.array(NO).mean(0)[-1] / SIG2, np.array(DE).mean(0)[-1] / SIG2))
except Exception as e:
    print("Fig 2 FAILED:", e)

# ---------------------------------------------------------------- Fig 3
try:
    r = load("pixel_mnist_recursion.jsonl")
    UN = [r[k]["traj"] for k in r if k.startswith("unfixed_")]
    FX = [r[k]["traj"] for k in r if k.startswith("fixed_")]
    true = r["TRUE_baseline"]["offman"]
    g = np.arange(len(UN[0]))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    band(ax, g, UN, C_BAD, "unfixed recursion  (collapse)")
    band(ax, g, FX, C_JMP, "+ local anchor  (ours)")
    ax.axhline(true, color=C_TRUE, ls=":", lw=1.6)
    ax.text(len(g) - 1, true - 0.08, "true-data baseline %.2f" % true, ha="right",
            va="top", color=C_TRUE, fontsize=10)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("off-manifold distance to real MNIST")
    ax.set_title("Real pixels (8×8 MNIST): anchored stays near the manifold")
    ax.legend(loc="center right", framealpha=0.95); ax.set_xticks(g)
    ax.set_ylim(1.0, 3.4)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, "fig3_pixel_mnist.png"))
    plt.close(fig)
    print("Fig 3 OK  (unfixed %.2f, fixed %.2f, true %.2f)"
          % (np.array(UN).mean(0)[-1], np.array(FX).mean(0)[-1], true))
except Exception as e:
    print("Fig 3 FAILED:", e)

# ---------------------------------------------------------------- Fig 4
try:
    kbar = np.array([2, 3, 3.9, 5, 7, 10.0])
    phi = np.array([14.1, 6.28, 3.82, 2.44, 1.43, 1.00])   # Phi_det/sigma^2, deficit_floor_law.py
    kk = np.linspace(1.7, 10.5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(kbar, phi, "o", color=C_STD, ms=8,
            label="$\\Phi_{\\rm det}/\\sigma^2$ (ODE at $\\sigma$=0.05, no fit)")
    C0 = (1 + 3 * np.exp(-4.0)) / 8  # exact sigma->0 constant (1+3e^-4)/8 = 0.1319
    ax.plot(kk, (C0 / kk ** 2) / SIG2, "--", color=C_DET,
            label="exact $\\sigma\\to0$ law  $\\frac{1+3e^{-4}}{8}\\,\\bar\\kappa^{-2}$")
    ax.axhline(1.0, color=C_TRUE, ls=":", lw=1.5)
    ax.text(10.3, 1.08, "$\\sigma^2$", color=C_TRUE, fontsize=11, ha="right")
    ax.plot([3.9], [3.82], "*", color=C_BAD, ms=18, zorder=5,
            label="measured net $\\bar\\kappa\\!=\\!3.9\\to3.8\\sigma^2$")
    ax.annotate("measured single-pass floor\n(zero-fit prediction)", xy=(3.9, 3.82),
                xytext=(5.2, 8), fontsize=10, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.4))
    ax.set_yscale("log"); ax.set_xlabel("network normal-slope ceiling  $\\bar\\kappa$")
    ax.set_ylabel("floor  $\\Phi_{\\rm det}/\\sigma^2$")
    ax.set_title("No finite network reaches $\\sigma^2$: the capacity floor $\\Phi_{\\rm det}(\\bar\\kappa)$")
    ax.legend(loc="upper right", framealpha=0.95); ax.set_ylim(0.7, 30)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, "fig4_capacity_floor.png"))
    plt.close(fig)
    print("Fig 4 OK")
except Exception as e:
    print("Fig 4 FAILED:", e)

print("figures in", OUTD)
