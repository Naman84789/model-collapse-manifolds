"""
The (D-1) coefficient, tested where it is 2 instead of 1.

eq. (varode-curved) says the curvature correction is Cov(h, Lap h) with Lap h = (D-1)/r
for a sphere of radius R in R^D. The ring tube (D=2) tests the coefficient 1. S^2 in R^3
tests the coefficient 2 -- a parameter-free doubling, on a geometry with a different net,
a different radial law and a different ambient dimension.

Same derivative-free estimator as curved_ode_onestep.py: frozen ensemble, one Euler step,
paired + antithetic + replicated, so the O(sqrt dtau) sampling term cancels exactly.

Reported per (t, dtau):
  ratio     (meas - flat) / (exact - flat)     -> 1 if the identity holds
  ratio_D1  the same with (D-1) forced to 1    -> 1 if the coefficient were really 1

If ratio ~ 1 and ratio_D1 ~ 2, the (D-1) scaling is confirmed and not fitted.

Run:  python curved_ode_onestep_sphere.py  ->  curved_ode_onestep_sphere.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "curved_ode_onestep_sphere.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0, KGRID = 1e-4, 1500
NSAMP = 150000
MREP = 48
PROBE_TS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
DTAUS = [1e-3, 1e-4, 1e-5]
DAMB = 3                                   # S^2 in R^3, so (D-1) = 2


def temb(t, dim=32):
    f = torch.exp(torch.linspace(0, 5, dim // 2)).to(t)
    return torch.cat([torch.sin(t * f), torch.cos(t * f)], 1)


class Score(nn.Module):
    def __init__(self, D=DAMB, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))


def sphere(n, sig, rng):
    """verbatim from capacity_domains.data_sphere"""
    v = rng.normal(size=(n, DAMB)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * (R + sig * rng.normal(size=(n, 1)))).astype("float32")


def train(sig, seed=500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(sphere(N, sig, rng))
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, DAMB); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def onestep(net, x, t, dtau, seed):
    s = float(np.sqrt(1 - np.exp(-t)))
    with torch.no_grad():
        e = net(x, torch.full((len(x), 1), float(t)))
    b = 0.5 * x - e / s
    xn = x.numpy()
    r = np.maximum(np.linalg.norm(xn, axis=1), 1e-12)
    u = xn / r[:, None]
    V0 = float(r.var())
    g = torch.Generator().manual_seed(seed)
    s1 = s2 = 0.0; ntot = 0
    for _ in range(MREP):
        xi = torch.randn(x.shape, generator=g)
        for sgn in (1.0, -1.0):
            xp = x + b * dtau + np.sqrt(dtau) * sgn * xi
            rp = np.linalg.norm(xp.numpy(), axis=1).astype(np.float64)
            s1 += rp.sum(); s2 += (rp ** 2).sum(); ntot += len(rp)
    mean = s1 / ntot
    meas = ((s2 / ntot - mean * mean) - V0) / dtau
    ub = (u * b.numpy()).sum(1)
    flat = 2 * float(np.cov(r, ub)[0, 1]) + 1
    cov1 = float(np.cov(r, 1.0 / r)[0, 1])           # Cov(r, 1/r)
    return meas, flat, flat + (DAMB - 1) * cov1, flat + 1.0 * cov1, V0, cov1


if __name__ == "__main__":
    t0_ = time.time(); out = []
    net = train(0.05)
    torch.manual_seed(11); x = torch.randn(NSAMP, DAMB)
    ts = np.geomspace(T_MAX, T0, KGRID + 1)
    want = {min(range(KGRID), key=lambda i: abs(ts[i] - p)) for p in PROBE_TS}
    print(f"S^2 in R^3, sigma=0.05.  (D-1) = {DAMB-1}.  NSAMP={NSAMP}, MREP={MREP} pairs.")
    print(f"  {'t':>8} {'dtau':>7} {'measured':>10} {'flat':>10} {'resid':>10}"
          f" {'curv(D-1=2)':>12} {'ratio':>7} {'ratio if D-1=1':>15}")
    for i in range(KGRID):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        s = np.sqrt(1 - np.exp(-t))
        if i in want:
            for dtau in DTAUS:
                m, fl, ex, ex1, V0, c1 = onestep(net, x, t, dtau, seed=2000 + i)
                res = m - fl; cv = ex - fl; cv1 = ex1 - fl
                rat = res / cv if abs(cv) > 1e-12 else float("nan")
                rat1 = res / cv1 if abs(cv1) > 1e-12 else float("nan")
                print(f"  {t:>8.4f} {dtau:>7.0e} {m:>10.5f} {fl:>10.5f} {res:>10.5f}"
                      f" {cv:>12.5f} {rat:>7.3f} {rat1:>15.3f}")
                out.append(dict(key=f"COSS_{t:.5f}_{dtau:g}", t=t, dtau=dtau,
                                V0=round(V0, 6), measured=round(m, 5), flat=round(fl, 5),
                                resid=round(res, 5), curv_D1_2=round(cv, 5),
                                curv_D1_1=round(cv1, 5),
                                ratio=None if np.isnan(rat) else round(rat, 4),
                                ratio_if_D1_is_1=None if np.isnan(rat1) else round(rat1, 4)))
        with torch.no_grad():
            e = net(x, torch.full((NSAMP, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    with open(OUT, "w") as f:
        for r_ in out:
            f.write(json.dumps(r_) + "\n")
    print(f"\nwrote {OUT}  ({time.time()-t0_:.0f}s)")
    print("identity holds  <=>  ratio ~ 1 and 'ratio if D-1=1' ~ 2.")
