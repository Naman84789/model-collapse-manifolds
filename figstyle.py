"""Shared publication style for all paper figures.

Principles (after AndreyChurkin/BeautifulFigures): vector output, fonts matched to the
manuscript (Computer Modern), one small muted palette used with fixed semantics across
every figure, frameless legends, no in-figure claim titles (captions carry the message),
figures designed at their final printed size so text is exactly 8-9 pt on the page.

Semantic palette (identical meaning in every figure):
  RED    collapse / unanchored arm        BLUE   ours (std sampler) / main theory curve
  GREEN  ours (jump sampler) / anchored   AMBER  secondary (deterministic horn, (U) law)
  GRAY   true-data baselines and reference lines
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREEN, AMBER, GRAY = "#b0413e", "#2e5a87", "#3e7c59", "#c87d2f", "#5b5b5b"
LIGHT = dict(red="#f3e4e3", blue="#e6edf4", green="#e5efe9", gray="#f2f2f2")

D = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(D, "paper", "figs")
SCRATCH = os.path.join(D, "figures")
for p in (FIGD, SCRATCH):
    os.makedirs(p, exist_ok=True)


def use_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "lines.linewidth": 1.6,
        "figure.dpi": 200,
        "savefig.dpi": 300,
    })


def save(fig, name):
    """Write name.pdf + name.png to paper/figs and name.png to figures/ (scratch)."""
    fig.savefig(os.path.join(FIGD, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIGD, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(SCRATCH, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def band(ax, x, T, color, label, lw=1.6, ls="-", z=3):
    """Mean line with +/- s.d. band."""
    import numpy as np
    T = np.asarray(T)
    m, s = T.mean(0), T.std(0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label, zorder=z)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0, zorder=z - 1)
    return m
