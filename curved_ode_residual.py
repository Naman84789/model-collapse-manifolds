"""
Does the variance ODE hold on CURVED data, and if not, is the missing term the radial
Ito correction?

Lemma 1 is exact in a normal coordinate that evolves as a plain Ito diffusion. The
radial coordinate does not: for x in R^D with r=|x|,
    dr = <u,f> dt + <u,dW> + (D-1)/(2r) dt,
and the last term is absent from the flat-tube derivation. Linearising it,
(D-1)/(2(R+h)) ~ (D-1)/(2R) - (D-1)h/(2R^2), whose slope contributes an extra
    -(D-1)/R^2 * V
to Var'. Prediction: measuring the ODE residual along the sampler trajectory gives ~0
for a flat normal coordinate and ~ -(D-1)V/R^2 for a radial one.

If confirmed, the corrected envelope is obtained by integrating
    -dV/dt = (1 - (D-1)/R^2 - 2 khat/s) V + 1
which is a STRICTLY SMALLER floor than the flat one, and therefore a bound that a
measured floor can satisfy where the flat bound fails.

Run:  python curved_ode_residual.py  ->  curved_ode_residual.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "curved_ode_residual.jsonl")
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0, KGRID = 1e-4, 4000            # fine grid: we differentiate V(t) numerically
NSAMP = 200000


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


def trace(net, geom, sig, seed=11):
    """Walk the sampler, recording V(t) and khat(t) in the geometry's normal coord."""
    torch.manual_seed(seed); x = torch.randn(NSAMP, 2)
    ts = np.geomspace(T_MAX, T0, KGRID + 1)
    rec = []
    for i in range(KGRID):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((NSAMP, 1), t))
        xn = x.numpy(); en = e.numpy()
        if geom == "ring":
            rad = np.linalg.norm(xn, axis=1); uh = xn / rad[:, None]
            h = rad; er = (en * uh).sum(1)
        else:
            h = xn[:, 1]; er = en[:, 1]
        hc = h - h.mean(); V = float(hc.var())
        kh = float(np.cov(hc, er)[0, 1] / max(V, 1e-12))
        rec.append((t, V, kh, float(h.mean())))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return np.array(rec), ts


if __name__ == "__main__":
    t0_ = time.time(); out = []
    for geom in ("seg", "ring"):
        for sig in (0.05, 0.13):
            net = train(geom, sig)
            rec, ts = trace(net, geom, sig)
            t, V, kh, mean = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
            # measured -dV/dt (t decreasing) vs the flat-ODE prediction
            dV = np.diff(V) / np.diff(ts[:KGRID + 1])[:len(V) - 1]
            meas = -dV                      # since t decreases, this is -dV/dt
            s_ = np.sqrt(1 - np.exp(-t[:-1]))
            pred_flat = (1 - 2 * kh[:-1] / s_) * V[:-1] + 1
            resid = meas - pred_flat
            # predicted curvature term: -(D-1)/R^2 * V  with D=2 -> -V/R^2, using mean radius
            curv = -V[:-1] / np.maximum(mean[:-1], 1e-9) ** 2 if geom == "ring" else 0 * V[:-1]
            band = (t[:-1] > 1e-3) & (t[:-1] < 1.0)
            rr = float(np.median(resid[band]))
            cc = float(np.median(curv[band])) if geom == "ring" else 0.0
            print(f"{geom:>5} sig={sig:.2f}: median ODE residual over t in (1e-3,1) = "
                  f"{rr:+.4f}   predicted curvature term = {cc:+.4f}   "
                  f"ratio = {rr/cc if cc else float('nan'):.2f}")
            out.append(dict(key=f"COR_{geom}_{sig}", geom=geom, sig=sig,
                            median_residual=round(rr, 5),
                            predicted_curv_term=round(cc, 5),
                            ratio=round(rr / cc, 3) if cc else None))
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}  ({time.time()-t0_:.0f}s)")
    print("expect: seg residual ~0; ring residual ~ predicted curvature term (ratio ~1)")
