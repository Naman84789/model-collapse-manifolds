"""
poolwidth_probe.py re-run with the ceiling measured under the SAMPLER's own law.

The published probe (poolwidth_probe.kbar_fit) draws its points from the forward
process, so it estimates the regression slope under p_t. Theorem 2 is stated on

    khat(t) = Cov(X_t, eps_hat(X_t,t)) / Var(X_t),   X_t the sampler's ensemble,

and for a nonlinear estimator the two differ. This script trains the same networks,
measures BOTH ceilings on each, refits the degradation law kbar(w) under each, and
propagates each fit through the deterministic fixed point so the recursion floor can
be compared against the quantity the theorem actually constrains.

Nothing here overwrites poolwidth_probe.jsonl.

Run:  python poolwidth_probe_samplerlaw.py
Writes poolwidth_probe_samplerlaw.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "poolwidth_probe_samplerlaw.jsonl")

# protocol identical to poolwidth_probe.py
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0_SAMPLE, KSTEPS = 0.005, 200
PROBE_TS = [0.1, 0.05, 0.02, 0.01, 0.005]
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
SEEDS = [0, 1, 2]
SIGMA_REF, LAM = 0.05, 0.5
SIG2_REF = SIGMA_REF ** 2


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


def train(sig, seed):
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


def _slope(xn, er):
    rad = np.linalg.norm(xn, axis=1)
    h = rad - rad.mean()
    return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))


def kbar_pt(net, sig):
    """PUBLISHED: probe points from the forward process."""
    ks = []
    for t in PROBE_TS:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        rng = np.random.default_rng(7)
        th = rng.uniform(0, 2 * np.pi, 4000); rr = a * (R + sig * rng.normal(size=4000))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(4000, 2))
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((4000, 1), float(t))).numpy() * uh).sum(1)
        ks.append(_slope(x, er))
    return max(ks)


def sample_and_kbar(net, n, seed):
    """Run the paper's sampler; read the slope off the live ensemble in flight."""
    torch.manual_seed(seed); x = torch.randn(n, 2)
    ts = np.geomspace(T_MAX, T0_SAMPLE, KSTEPS + 1)
    want = {min(range(KSTEPS), key=lambda i: abs(ts[i] - p)) for p in PROBE_TS}
    ks = []
    for i in range(KSTEPS):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        if i in want:
            xn = x.numpy(); rad = np.linalg.norm(xn, axis=1)
            ks.append(_slope(xn, (e.numpy() * (xn / rad[:, None])).sum(1)))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    v = float(np.linalg.norm(x.numpy(), axis=1).var())
    return max(ks), v


def Phi_det(w, kbar, t0=1e-4, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0:
            V = 1e-12
    return V


def fixed_point(kbar_fn, t0=1e-4, damp=0.5):
    v = 0.01
    for _ in range(300):
        w = LAM * SIG2_REF + (1 - LAM) * v
        vn = Phi_det(w, float(max(kbar_fn(w), 0.3)), t0)
        if abs(vn - v) < 1e-9:
            v = vn; break
        v = damp * v + (1 - damp) * vn
    return v


if __name__ == "__main__":
    t_start = time.time()
    print(f"{'sig':>6} {'w':>8} {'kbar_pt':>18} {'kbar_samp':>18} {'ratio':>7} {'v_out':>9}")
    recs = []
    for sig in SIGS:
        kpt, ksa, vs = [], [], []
        for sd in SEEDS:
            net = train(sig, 500 + sd)
            kpt.append(kbar_pt(net, sig))
            k, v = sample_and_kbar(net, 4000, 900 + sd)
            ksa.append(k); vs.append(v)
        w = sig * sig
        r = dict(key=f"PWS_sig{sig}", sig=sig, w=round(w, 6),
                 kbar_pt=round(float(np.mean(kpt)), 4),
                 kbar_pt_range=[round(min(kpt), 3), round(max(kpt), 3)],
                 kbar_sampler=round(float(np.mean(ksa)), 4),
                 kbar_sampler_range=[round(min(ksa), 3), round(max(ksa), 3)],
                 ratio=round(float(np.mean(ksa) / np.mean(kpt)), 4),
                 v_out=round(float(np.mean(vs)), 6))
        recs.append(r)
        print(f"{sig:6.2f} {w:8.5f} {np.mean(kpt):7.3f} [{min(kpt):.2f},{max(kpt):.2f}] "
              f"{np.mean(ksa):8.3f} [{min(ksa):.2f},{max(ksa):.2f}] "
              f"{np.mean(ksa)/np.mean(kpt):7.3f} {np.mean(vs):9.5f}")

    ws = np.array([r["w"] for r in recs]); sq = np.sqrt(ws)
    print()
    print("degradation law  kbar(w) = A + B sqrt(w)   (published p_t fit: 4.4 - 11.6 sqrt(w))")
    fits = {}
    for tag in ("kbar_pt", "kbar_sampler"):
        y = np.array([r[tag] for r in recs])
        B, A = np.polyfit(sq, y, 1)
        resid = np.abs(y - (A + B * sq)).max()
        fits[tag] = (A, B)
        print(f"  {tag:13s}: kbar(w) = {A:6.3f} {B:+7.3f} sqrt(w)   max|resid| = {resid:.4f}")

    print()
    print("deterministic fixed point at sigma=0.05 (measured recursion floor ~11 sigma^2)")
    out = {}
    for tag, (A, B) in fits.items():
        v = fixed_point(lambda w: A + B * np.sqrt(w))
        out[tag] = v / SIG2_REF
        print(f"  using {tag:13s}: v* = {v/SIG2_REF:6.2f} sigma^2")

    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(dict(key="FIT", fits={k: [round(a, 4), round(b, 4)]
                                                 for k, (a, b) in fits.items()},
                                fixed_point_s2={k: round(v, 3) for k, v in out.items()})) + "\n")
    print(f"\nwrote {OUT}   ({time.time()-t_start:.0f}s)")
