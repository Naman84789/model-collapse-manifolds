"""Paper figures from the verified jsonl runs. Each figure is self-contained and guarded
so a missing input can't kill the others. Outputs vector PDFs (+ PNG twins) to
paper/figs/ via the shared publication style in figstyle.py.

Fig 1  head-to-head     : annealed truncation vs our anchored fix (rvar/sigma^2)
Fig 2  two-horned trap  : deterministic->point, stochastic->fat, anchored->sigma^2
Fig 3  pixel MNIST      : unanchored drift vs anchored vs true-data baseline
Fig 4  capacity floor   : Phi_det(kbar) envelope + the interventional test
                          (five training protocols from ceiling_origin.jsonl; the
                          measured floor tracks the law and never crosses the bound)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

import figstyle as st

st.use_style()
D = st.D
SIG2 = 0.05 ** 2

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

# ---------------------------------------------------------------- Fig 1
try:
    r = load("head_to_head.jsonl")
    A = [r[k]["traj"] for k in r if k.startswith("A_")]
    B = [r[k]["traj"] for k in r if k.startswith("B_")]
    Cc = [r[k]["traj"] for k in r if k.startswith("C_")]
    g = np.arange(len(A[0]))
    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    st.band(ax, g, np.array(A) / SIG2, st.RED, "annealed truncation (prior remedy, no anchor)")
    st.band(ax, g, np.array(B) / SIG2, st.BLUE, "ours: standard sampler + anchor")
    st.band(ax, g, np.array(Cc) / SIG2, st.GREEN, "ours: jump sampler + anchor", ls="--")
    ax.axhline(1.0, color=st.GRAY, ls=":", lw=1.0, zorder=1)
    ax.text(0.2, 1.09, "true tube $\\sigma^2$", color=st.GRAY, fontsize=8, va="bottom")
    ax.set_yscale("log"); ax.set_ylim(0.7, 60)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("output variance / $\\sigma^2$")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42))
    ax.set_xticks(g[::2])
    ax.grid(axis="y", lw=0.5, alpha=0.18)
    st.save(fig, "fig1_head_to_head")
    print("Fig 1 OK  (annealed g19 = %.1f s2, ours = %.2f s2)"
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
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    st.band(ax, g, np.array(NO) / SIG2, st.RED,
            "stochastic sampler: stuck fat ($\\sim$11$\\sigma^2$)")
    mB = np.array(B)[:, :len(g)]
    st.band(ax, g, mB / SIG2, st.BLUE, "+ anchor: holds $\\sigma^2$")
    mDE = np.array(DE).mean(0) / SIG2
    ax.plot(g, np.clip(mDE, 3e-3, None), color=st.AMBER, lw=1.6,
            label="deterministic sampler: point collapse")
    ax.axhline(1.0, color=st.GRAY, ls=":", lw=1.0, zorder=1)
    ax.text(len(g) - 0.2, 1.25, "true tube $\\sigma^2$", color=st.GRAY,
            fontsize=8, ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(2.2e-3, 60)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("output variance / $\\sigma^2$")
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.42))
    ax.set_xticks(g[::2])
    ax.grid(axis="y", lw=0.5, alpha=0.18)
    st.save(fig, "fig2_two_horned_trap")
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
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    st.band(ax, g, UN, st.RED, "unanchored recursion (collapse)")
    st.band(ax, g, FX, st.GREEN, "+ local anchor (ours)")
    ax.axhline(true, color=st.GRAY, ls=":", lw=1.0)
    ax.text(len(g) - 1, true - 0.05, "true-data baseline %.2f" % true, ha="right",
            va="top", color=st.GRAY, fontsize=8)
    ax.set_xlabel("generation of self-consuming training")
    ax.set_ylabel("off-manifold distance to real MNIST")
    ax.legend(loc="center right")
    ax.set_xticks(g)
    ax.set_ylim(1.0, 3.4)
    ax.grid(axis="y", lw=0.5, alpha=0.18)
    st.save(fig, "fig3_pixel_mnist")
    print("Fig 3 OK  (unfixed %.2f, fixed %.2f, true %.2f)"
          % (np.array(UN).mean(0)[-1], np.array(FX).mean(0)[-1], true))
except Exception as e:
    print("Fig 3 FAILED:", e)

# ---------------------------------------------------------------- Fig 4
try:
    # finite-sigma ODE floor values, deficit_floor_law.py (no fit)
    kbar = np.array([2, 3, 3.9, 5, 7, 10.0])
    phi = np.array([14.1, 6.28, 3.82, 2.44, 1.43, 1.00])
    kk = np.linspace(1.7, 10.5, 200)
    # interventional test, ceiling_origin.jsonl: five training protocols,
    # (measured kbar, measured single-pass floor at t0=1e-4) per arm A..E
    co = load("ceiling_origin.jsonl")
    if co:
        ik = np.array([co[k]["kbar"] for k in sorted(co)])
        im = np.array([co[k]["floor_meas_sig2"] for k in sorted(co)])
    else:  # logged values, for a repo checkout without the jsonl
        ik = np.array([3.734, 6.678, 3.476, 7.466, 6.431])
        im = np.array([7.391, 3.719, 7.793, 4.894, 2.456])

    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    C0 = (1 + 3 * np.exp(-4.0)) / 8  # exact sigma->0 constant (1+3e^-4)/8
    ax.plot(kk, (C0 / kk ** 2) / SIG2, "--", color=st.AMBER, lw=1.4,
            label="exact $\\sigma\\to0$ law $\\;\\frac{1+3e^{-4}}{8}\\,\\bar\\kappa^{-2}$")
    # envelope floor as a connecting line + markers, so "above the bound" is
    # visually rigorous (every measured floor sits above this curve).
    order = np.argsort(kbar)
    ax.plot(kbar[order], phi[order], "-o", color=st.BLUE, ms=5, lw=1.0,
            label="envelope floor (ODE at $\\sigma$=0.05, no fit)")
    ax.plot([3.9], [3.82], "*", color=st.BLUE, ms=13, zorder=5,
            label="envelope bound at measured $\\bar\\kappa$=3.9")
    ax.plot(ik, im, "o", color=st.RED, ms=5, mfc="none", mew=1.4, zorder=4,
            label="measured floors, five training protocols")
    ax.axhline(1.0, color=st.GRAY, ls=":", lw=1.0)
    ax.text(10.3, 1.07, "$\\sigma^2$", color=st.GRAY, fontsize=8, ha="right")
    ax.annotate("raising $\\bar\\kappa$ lowers the floor;\nevery point above the bound",
                xy=(10.9, 6.6), fontsize=8, color=st.RED, ha="right", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("network normal-slope ceiling $\\bar\\kappa$")
    ax.set_ylabel("single-pass floor / $\\sigma^2$")
    ax.legend(loc="upper right")
    ax.set_ylim(0.7, 40)
    ax.grid(axis="y", lw=0.5, alpha=0.18)
    st.save(fig, "fig4_capacity_floor")
    print("Fig 4 OK  (interventional points: %d)" % len(ik))
except Exception as e:
    print("Fig 4 FAILED:", e)

print("figures in", st.FIGD)
