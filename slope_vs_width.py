"""
WHY do the two probes disagree, and only on curved geometries?

Both probes regress the network's normal eps-component on the normal coordinate. They
differ in the ENSEMBLE they regress over: p_t is a tube of width gamma(t) about the
contracted manifold; the sampler's own ensemble is provably fatter (that is the floor).

Hypothesis H: the measured slope is a decreasing function of the ensemble width, and the
sensitivity d kappa / d width is controlled by curvature -- zero for a flat normal
coordinate, positive for a curved one.

This isolates it: hold t, the network, and the ensemble CENTRE fixed, and sweep only the
ensemble width. Any dependence is then a property of the probe, not of the sampler.

Run:  python slope_vs_width.py  ->  slope_vs_width.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "slope_vs_width.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
SIG = 0.05
PROBE_TS = [0.1, 0.05, 0.02, 0.01, 0.005]
WIDTH_MULT = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]     # multiples of the p_t normal width


def temb(t, dim=32):
    f = torch.exp(torch.linspace(0, 5, dim // 2)).to(t)
    return torch.cat([torch.sin(t * f), torch.cos(t * f)], 1)


class Score(nn.Module):
    def __init__(self, D=2, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))


def ring(n, sig, rng):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")


def seg(n, sig, rng):
    L = 2 * np.pi * R
    x = rng.uniform(-L / 2, L / 2, n); y = sig * rng.normal(size=n)
    return np.stack([x, y], 1).astype("float32")


GEOM = {"ring": ring, "seg": seg}


def train(geom, sig, seed=500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(GEOM[geom](N, sig, rng))
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def slope_at_width(net, geom, t, mult, n=20000, seed=7):
    """Ensemble centred on the contracted manifold, normal width scaled by `mult`.

    The p_t ensemble has normal sd gamma = sqrt(a^2 sig^2 + s^2). We build the SAME
    centre and shape but with normal sd = mult * gamma, so only the width changes.
    """
    rng = np.random.default_rng(seed)
    a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
    gam = np.sqrt(a * a * SIG * SIG + s * s)
    eta = mult * gam * rng.normal(size=n)          # normal offset at the chosen width
    if geom == "ring":
        th = rng.uniform(0, 2 * np.pi, n)
        rad = a * R + eta
        x = np.stack([rad * np.cos(th), rad * np.sin(th)], 1).astype("float32")
        with torch.no_grad():
            e = net(torch.tensor(x), torch.full((n, 1), float(t))).numpy()
        uh = x / np.linalg.norm(x, axis=1)[:, None]
        er = (e * uh).sum(1)
        h = np.linalg.norm(x, axis=1) - a * R
    else:
        L = 2 * np.pi * R
        xs = rng.uniform(-L / 2, L / 2, n)
        x = np.stack([xs, eta], 1).astype("float32")
        with torch.no_grad():
            e = net(torch.tensor(x), torch.full((n, 1), float(t))).numpy()
        er = e[:, 1]; h = x[:, 1]
    return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))


if __name__ == "__main__":
    t0_ = time.time(); recs = []
    print("measured slope vs ENSEMBLE WIDTH (multiples of the p_t width), fixed net & t")
    for geom in ("seg", "ring"):
        net = train(geom, SIG)
        print(f"\n--- {geom} ---")
        print(f"{'t':>7} " + "".join(f"{'x%.1f' % m:>9}" for m in WIDTH_MULT) + f"{'fall':>9}")
        for t in PROBE_TS:
            ks = [slope_at_width(net, geom, t, m) for m in WIDTH_MULT]
            fall = 1 - ks[-1] / ks[0]
            print(f"{t:7.3f} " + "".join(f"{k:9.3f}" for k in ks) + f" {100*fall:7.1f}%")
            recs.append(dict(key=f"SW_{geom}_{t}", geom=geom, t=t,
                             mult=WIDTH_MULT, kappa=[round(k, 4) for k in ks],
                             fall_pct=round(100 * fall, 2)))
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}  ({time.time()-t0_:.0f}s)")
    print("H predicts: seg ~flat in width, ring falls with width.")
