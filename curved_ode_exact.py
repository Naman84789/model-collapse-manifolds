"""
Does the measured ring residual equal the EXACT curvature term (D-1) Cov(r, 1/r)?

radial_ode_identity.py proves, to 50 digits and symbolically, that in reverse time the
radial variance obeys

    -dV/dt = (1 - 2 khat/s) V + 1 + (D-1) Cov(r, 1/r),
    Cov(r, 1/r) = 1 - E[r]E[1/r] = -1/2 E[(r-r')^2/(r r')]   (r,r' iid)  <= 0.

curved_ode_residual.py compared the measured residual against the LINEARISATION
-(D-1) V / E[r]^2 and found it 5-6x too small. This re-runs the same nets (same seeds,
same architecture, same training) and compares the residual against the exact term.

Prediction: residual / exact ~ 1 pointwise in t; residual / linearised ~ 5-6.

Also recorded, to show WHY the linearisation fails: the small-radius quantiles of the
sampler's own ensemble. Cov(r,1/r) is dominated by small r, so the expansion about the
mean radius has no reason to work once the ensemble is fat -- which is exactly the
regime the floor puts it in.

Run:  python curved_ode_exact.py  ->  curved_ode_exact.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "curved_ode_exact.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0, KGRID = 1e-4, 1500
NSAMP = 100000
DDIM = 2                                   # ambient dimension of these two geometries


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
    """Byte-identical to curved_ode_residual.py / slope_vs_width.py, so the nets match."""
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


def trace(net, geom, seed=11):
    torch.manual_seed(seed); x = torch.randn(NSAMP, 2)
    ts = np.geomspace(T_MAX, T0, KGRID + 1)
    prng = np.random.default_rng(3)
    rec = []
    for i in range(KGRID):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((NSAMP, 1), t))
        xn = x.numpy(); en = e.numpy()
        if geom == "ring":
            r = np.linalg.norm(xn, axis=1)
            er = (en * (xn / r[:, None])).sum(1)
            h = r
        else:
            h = xn[:, 1]; er = en[:, 1]
        V = float(h.var())
        kh = float(np.cov(h - h.mean(), er)[0, 1] / max(V, 1e-12))
        if geom == "ring":
            rr = np.maximum(r, 1e-12)
            m = float(rr.mean()); inv = float((1.0 / rr).mean())
            # two-point form, as an independent estimate of the same covariance
            perm = prng.permutation(NSAMP)
            tp = float(-0.5 * np.mean((rr - rr[perm]) ** 2 / (rr * rr[perm])))
            q = np.quantile(rr, [0.0001, 0.001, 0.01, 0.05])
            rec.append((t, V, kh, m, inv, tp, *q))
        else:
            rec.append((t, V, kh, float(h.mean()), 0.0, 0.0, 0, 0, 0, 0))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return np.array(rec), ts


def report(geom, sig, rec, ts, lo, hi, out):
    t, V, kh, m, inv, tp = (rec[:, i] for i in range(6))
    dV = np.diff(V) / np.diff(ts[:KGRID + 1])[:len(V) - 1]
    meas = -dV
    s_ = np.sqrt(1 - np.exp(-t[:-1]))
    flat = (1 - 2 * kh[:-1] / s_) * V[:-1] + 1
    resid = meas - flat
    if geom == "ring":
        exact = (DDIM - 1) * (1.0 - m[:-1] * inv[:-1])
        exact_tp = (DDIM - 1) * tp[:-1]
        lin = -(DDIM - 1) * V[:-1] / m[:-1] ** 2
    else:
        exact = exact_tp = lin = 0.0 * V[:-1]
    band = (t[:-1] > lo) & (t[:-1] < hi)
    med = lambda a: float(np.median(a[band]))
    row = dict(key=f"CE_{geom}_{sig}_{lo:g}_{hi:g}", geom=geom, sig=sig,
               band=[lo, hi], n=int(band.sum()),
               median_residual=round(med(resid), 5),
               median_exact=round(med(exact), 5),
               median_exact_twopoint=round(med(exact_tp), 5),
               median_linearised=round(med(lin), 5),
               median_ratio_resid_over_exact=round(med(resid / np.where(exact != 0, exact, np.nan)), 4)
               if geom == "ring" else None,
               median_ratio_resid_over_lin=round(med(resid / np.where(lin != 0, lin, np.nan)), 4)
               if geom == "ring" else None,
               median_V=round(med(V[:-1]), 5), median_mean_r=round(med(m[:-1]), 4))
    if geom == "ring":
        row["radius_quantiles_0.01_0.1_1_5pct"] = [round(med(rec[:-1, 6 + j]), 4) for j in range(4)]
    out.append(row)
    print(f"  {geom:>4} sig={sig:.2f}  t in ({lo:g},{hi:g})  n={int(band.sum())}")
    print(f"      residual {med(resid):+.5f}   EXACT {med(exact):+.5f}"
          f"   two-point {med(exact_tp):+.5f}   linearised {med(lin):+.5f}")
    if geom == "ring":
        pe = 100 * med(exact) / med(resid) if med(resid) else float('nan')
        pl = 100 * med(lin) / med(resid) if med(resid) else float('nan')
        print(f"      exact explains {pe:6.1f}% of the residual;"
              f" linearisation explains {pl:6.1f}%")
        print(f"      median r quantiles [0.01%,0.1%,1%,5%] = "
              f"{[round(med(rec[:-1,6+j]),3) for j in range(4)]}   median E[r]={med(m[:-1]):.3f}")
    print()


if __name__ == "__main__":
    t0_ = time.time(); out = []
    for geom, sig in (("ring", 0.05), ("ring", 0.13), ("seg", 0.05)):
        print(f"=== {geom} sigma={sig} ===")
        net = train(geom, sig)
        rec, ts = trace(net, geom)
        report(geom, sig, rec, ts, 1e-3, 1.0, out)       # the original band
        report(geom, sig, rec, ts, 1e-4, 1e-2, out)      # near the floor
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}  ({time.time()-t0_:.0f}s)")
