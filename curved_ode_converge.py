"""
Is the ring's ODE residual a property of the SDE, or of the Euler step?

radial_ode_identity.py proves, symbolically and to 50 digits, that the continuous-time
residual is exactly (D-1) Cov(r, 1/r). curved_ode_exact.py measured, on the trained ring
net, a residual ~3x that term over t in (1e-3, 1) -- but the SAME measurement gave -0.0624
at KGRID=4000 and -0.0405 at KGRID=1500. A 35% move under a change of step count is not a
property of the SDE. Euler-Maruyama has an O(dt) bias in dV/dt here, and the drift carries
epshat/s with s ~ sqrt(t), so the left-endpoint evaluation is worst exactly where the
residual was read.

This sweeps the step count on ONE net (ring, sigma=0.05, seed 500 -- the same net) and
asks whether residual/exact -> 1 as dt -> 0. Two estimators per grid:
  left  : RHS at t_i                    (what the earlier probes used)
  trap  : RHS averaged over t_i, t_i+1  (removes the leading O(dt) midpoint bias)

If the ratio converges to 1, the identity is confirmed on the network and the earlier
"5-6x" was a discretisation artifact. If it converges to something else, there is a term
neither the flat ODE nor the curvature correction contains, and I want to know that.

Run:  python curved_ode_converge.py  ->  curved_ode_converge.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "curved_ode_converge.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0 = 1e-4
NSAMP = 50000
GRIDS = [1000, 2000, 4000, 8000]
DDIM = 2
SIG = 0.05


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


def train(sig, seed=500):
    """Identical to curved_ode_residual.py / curved_ode_exact.py -- same net."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(ring(N, sig, rng))
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def trace(net, kgrid, seed=11):
    torch.manual_seed(seed); x = torch.randn(NSAMP, 2)
    ts = np.geomspace(T_MAX, T0, kgrid + 1)
    rec = []
    for i in range(kgrid):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((NSAMP, 1), t))
        xn = x.numpy(); en = e.numpy()
        r = np.maximum(np.linalg.norm(xn, axis=1), 1e-12)
        er = (en * (xn / r[:, None])).sum(1)
        V = float(r.var())
        kh = float(np.cov(r - r.mean(), er)[0, 1] / max(V, 1e-12))
        rec.append((t, V, kh, float(r.mean()), float((1.0 / r).mean())))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return np.array(rec), ts


def analyse(rec, ts, kgrid, lo, hi):
    t, V, kh, m, inv = (rec[:, i] for i in range(5))
    meas = -np.diff(V) / np.diff(ts[:kgrid + 1])[:len(V) - 1]
    s = np.sqrt(1 - np.exp(-t))
    rhs = (1 - 2 * kh / s) * V + 1
    exact_all = (DDIM - 1) * (1.0 - m * inv)
    lin_all = -(DDIM - 1) * V / m ** 2
    band = (t[:-1] > lo) & (t[:-1] < hi)
    med = lambda a: float(np.median(a[band]))
    o = {}
    for name, R_, E_, L_ in (("left", rhs[:-1], exact_all[:-1], lin_all[:-1]),
                             ("trap", 0.5 * (rhs[:-1] + rhs[1:]),
                              0.5 * (exact_all[:-1] + exact_all[1:]),
                              0.5 * (lin_all[:-1] + lin_all[1:]))):
        res = meas - R_
        o[name] = dict(residual=round(med(res), 5), exact=round(med(E_), 5),
                       lin=round(med(L_), 5),
                       pct_exact=round(100 * med(E_) / med(res), 1) if med(res) else None,
                       pct_lin=round(100 * med(L_) / med(res), 1) if med(res) else None)
    o["n"] = int(band.sum())
    return o


if __name__ == "__main__":
    t0_ = time.time()
    net = train(SIG)
    out = []
    print(f"ring sigma={SIG}, one net (seed 500), NSAMP={NSAMP}.")
    print("percentages = how much of the measured residual the term accounts for.\n")
    for lo, hi in ((1e-3, 1.0), (1e-4, 1e-2)):
        print(f"--- band t in ({lo:g}, {hi:g}) ---")
        print(f"  {'KGRID':>7} {'est':>5} {'residual':>10} {'exact':>10} {'lin':>10}"
              f" {'%exact':>8} {'%lin':>7}")
        for kg in GRIDS:
            rec, ts = trace(net, kg)
            a = analyse(rec, ts, kg, lo, hi)
            for est in ("left", "trap"):
                d = a[est]
                print(f"  {kg:>7} {est:>5} {d['residual']:>10.5f} {d['exact']:>10.5f}"
                      f" {d['lin']:>10.5f} {str(d['pct_exact']):>8} {str(d['pct_lin']):>7}")
            out.append(dict(key=f"COC_{kg}_{lo:g}_{hi:g}", kgrid=kg, band=[lo, hi],
                            nsamp=NSAMP, **a))
            print()
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}  ({time.time()-t0_:.0f}s)")
    print("if %exact -> 100 as KGRID grows, the identity holds on the network too.")

# OUTCOME (2026-08-28): the residual did NOT converge to the exact term -- it moved from
# -0.041 at KGRID=1000 to -0.050 at 2000, AWAY from it, and the predicted term itself moved
# 10.6% because the trajectory changes with the step count. Diagnosis: this whole family of
# estimators forms (V[i+1]-V[i])/dt, so sampling error in V is divided by dt and diverges as
# dt -> 0. Superseded by curved_ode_onestep.py, which freezes the ensemble and takes ONE
# paired antithetic step; that estimator confirms the identity to a ratio of 0.99.
# Kept as the record of why trajectory-differencing must not be used for this quantity.
