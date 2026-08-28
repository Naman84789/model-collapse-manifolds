"""
Test the curved variance ODE on the real network WITHOUT differentiating a trajectory.

curved_ode_converge.py shows the trajectory-differencing residual GROWS as the grid
refines (-0.0411 at KGRID=1000, -0.0497 at 2000), moving away from the exact term rather
than toward it. That is the signature of the estimator, not of the SDE: meas = dV/dt is
formed as (V[i+1]-V[i])/dt, so any sampling error in V is divided by dt and diverges as
dt -> 0.

This measures the same quantity with the noise removed by construction. At a frozen t,
take the sampler's own ensemble and apply ONE Euler step, using:
  * PAIRING       -- the same base particles before and after, so Var(r) cancels;
  * ANTITHETIC    -- replicate each step with +xi and -xi, which cancels the O(sqrt dtau)
                     term 2 sqrt(dtau) Cov(r, zeta) exactly, the term that dominates the
                     error above;
  * REPLICATION   -- M noise replicates pooled, so what is left falls as 1/sqrt(M N).
Then compare (Var(r') - Var(r)) / dtau against the two predictions:
  flat   2 Cov(r, <u,b>) + 1                         [eq. (varode)]
  exact  2 Cov(r, <u,b> + (D-1)/(2r)) + 1            [eq. (varode-curved)]

Several dtau, to show the answer is not a step artifact. Flat SEG geometry as a control:
there the two predictions coincide and both must match.

Run:  python curved_ode_onestep.py  ->  curved_ode_onestep.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "curved_ode_onestep.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0, KGRID = 1e-4, 1500
NSAMP = 200000
MREP = 48                      # antithetic pairs per frozen time
PROBE_TS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
DTAUS = [1e-3, 1e-4, 1e-5]


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
    return np.stack([rng.uniform(-L / 2, L / 2, n),
                     sig * rng.normal(size=n)], 1).astype("float32")


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


def coord(geom, xn):
    """normal coordinate h and unit normal n"""
    if geom == "ring":
        r = np.maximum(np.linalg.norm(xn, axis=1), 1e-12)
        return r, xn / r[:, None]
    return xn[:, 1], np.tile(np.array([[0.0, 1.0]], dtype="float32"), (len(xn), 1))


def onestep(net, geom, x, t, dtau, seed):
    """Var of the one-step-pushed ensemble minus Var now, paired + antithetic."""
    s = float(np.sqrt(1 - np.exp(-t)))
    with torch.no_grad():
        e = net(x, torch.full((len(x), 1), float(t)))
    b = 0.5 * x - e / s                                     # reverse-time drift
    xn = x.numpy(); h, u = coord(geom, xn)
    V0 = float(h.var())
    g = torch.Generator().manual_seed(seed)
    s1 = s2 = 0.0; n_tot = 0
    for _ in range(MREP):
        xi = torch.randn(x.shape, generator=g)
        for sgn in (1.0, -1.0):                             # antithetic pair
            xp = x + b * dtau + np.sqrt(dtau) * sgn * xi
            hp, _ = coord(geom, xp.numpy())
            s1 += float(hp.sum()); s2 += float((hp.astype(np.float64) ** 2).sum())
            n_tot += len(hp)
    mean = s1 / n_tot
    V1 = s2 / n_tot - mean * mean
    meas = (V1 - V0) / dtau
    ub = (u * b.numpy()).sum(1)
    flat = 2 * float(np.cov(h, ub)[0, 1]) + 1
    if geom == "ring":
        lap = 1.0 / h                                       # (D-1)/r with D=2
        exact = flat + float(np.cov(h, lap)[0, 1])
    else:
        exact = flat
    return meas, flat, exact, V0


if __name__ == "__main__":
    t0_ = time.time(); out = []
    for geom, sig in (("ring", 0.05), ("seg", 0.05)):
        print(f"\n=== {geom} sigma={sig} ===")
        net = train(geom, sig)
        # walk the sampler, freezing the ensemble at each probe time
        torch.manual_seed(11); x = torch.randn(NSAMP, 2)
        ts = np.geomspace(T_MAX, T0, KGRID + 1)
        want = {min(range(KGRID), key=lambda i: abs(ts[i] - p)): p for p in PROBE_TS}
        print(f"  {'t':>8} {'dtau':>8} {'measured':>11} {'flat':>11} {'exact':>11}"
              f" {'meas-flat':>11} {'exact-flat':>11} {'ratio':>7}")
        for i in range(KGRID):
            t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
            s = np.sqrt(1 - np.exp(-t))
            if i in want:
                for dtau in DTAUS:
                    m, fl, ex, V0 = onestep(net, geom, x, t, dtau, seed=1000 + i)
                    rat = (m - fl) / (ex - fl) if abs(ex - fl) > 1e-12 else float("nan")
                    print(f"  {t:>8.4f} {dtau:>8.0e} {m:>11.5f} {fl:>11.5f} {ex:>11.5f}"
                          f" {m-fl:>11.5f} {ex-fl:>11.5f} {rat:>7.3f}")
                    out.append(dict(key=f"COS_{geom}_{t:.5f}_{dtau:g}", geom=geom, t=t,
                                    dtau=dtau, V0=round(V0, 6),
                                    measured=round(m, 5), flat=round(fl, 5),
                                    exact=round(ex, 5),
                                    resid=round(m - fl, 5),
                                    curv_term=round(ex - fl, 5),
                                    ratio=None if np.isnan(rat) else round(rat, 4)))
            with torch.no_grad():
                e = net(x, torch.full((NSAMP, 1), t))
            x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}  ({time.time()-t0_:.0f}s)")
    print("ring: ratio ~ 1 confirms the curvature term is the whole residual.")
    print("seg : exact == flat by construction, so meas-flat ~ 0 is the control.")
