"""
The interventional test, re-measured under the SAMPLER's own law.

ceiling_origin.py reads each arm's ceiling off the FORWARD process (profile_pt).
Theorem 2 is stated on the slope under the sampler's own ensemble, eq. (3). Since
the floor goes as kbar^-2, a lower sampler-law ceiling RAISES each arm's bound, so
the paper's claim -- every one of the five sits strictly above its own proven
bound -- has to be re-checked, not assumed.

The five arms are trained exactly as in ceiling_origin.py (same seeds, same
protocol switches), then each ceiling is measured BOTH ways on the same network.

Run:  python ceiling_origin_samplerlaw.py   ->  ceiling_origin_samplerlaw.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ceiling_origin_samplerlaw.jsonl")

# protocol identical to ceiling_origin.py
R, SIG, T_MAX, BATCH, N = 2.5, 0.05, 8.0, 512, 6000
T0, KGRID = 1e-4, 200
TS = np.geomspace(T_MAX, T0, KGRID + 1)
SIG2 = SIG * SIG
ARMS = ["A", "B", "C", "D", "E"]
SEEDS = [500, 501, 502]


class Score(nn.Module):
    def __init__(self, D=2, h=256, te=32, fmax_exp=5.0):
        super().__init__(); self.te = te
        self.register_buffer("freqs", torch.exp(torch.linspace(0, fmax_exp, te // 2)))
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def forward(self, x, t):
        emb = torch.cat([torch.sin(t * self.freqs), torch.cos(t * self.freqs)], 1)
        return self.net(torch.cat([x, emb], 1))


def ring(n, sig, rng):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")


def draw_t(n, band_frac):
    t_std = torch.rand(n, 1) * (T_MAX - 1e-3) + 1e-3
    if band_frac <= 0:
        return t_std
    nb = int(round(band_frac * n))
    lo, hi = np.log(2e-4), np.log(0.1)
    t_band = torch.exp(torch.rand(nb, 1) * (hi - lo) + lo)
    return torch.cat([t_band, t_std[nb:]], 0)


def train(arm, seed=500):
    steps = 8000 if arm == "E" else 2000
    fmax = 8.0 if arm in ("C", "D") else 5.0
    bf = 0.5 if arm in ("B", "D") else 0.0
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(ring(N, SIG, rng))
    net = Score(fmax_exp=fmax); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = draw_t(BATCH, bf)
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def _slope(xn, er):
    rad = np.linalg.norm(xn, axis=1); h = rad - rad.mean()
    return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))


def profile_pt(net, n=4000, seed=7):
    """PUBLISHED probe: points from the forward process."""
    rng = np.random.default_rng(seed); ks = []
    for t in TS[:-1]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        th = rng.uniform(0, 2 * np.pi, n); rr = a * (R + SIG * rng.normal(size=n))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(n, 2))
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((n, 1), float(t))).numpy() * uh).sum(1)
        ks.append(_slope(x, er))
    return max(ks)


def sample_and_kbar(net, n=4000, seed=900):
    """Sampler-law ceiling, read off the live ensemble; also returns the floor."""
    torch.manual_seed(seed); x = torch.randn(n, 2); ks = []
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        xn = x.numpy(); rad = np.linalg.norm(xn, axis=1)
        ks.append(_slope(xn, (e.numpy() * (xn / rad[:, None])).sum(1)))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return max(ks), float(np.linalg.norm(x.numpy(), axis=1).var())


def integrate(t0, kbar, sigma, K=400000, tstart=8.0):
    """deficit_floor_law.integrate, cap_mode='min'."""
    s2 = sigma * sigma
    ks = lambda t: (lambda a2, s: s / (a2 * s2 + s * s))(np.exp(-t), np.sqrt(1 - np.exp(-t)))
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        V = V + ((1 - 2 * min(ks(t), kbar) / s) * V + 1) * dt
        if V < 0:
            V = 1e-12
    return V


if __name__ == "__main__":
    t_start = time.time()
    # resumable: each arm is appended as it finishes, so a power cut costs one arm
    recs, done = [], set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); recs.append(r); done.add(r["arm"])
            except Exception:
                pass
    print("interventional arms under both probes (3 seeds each)")
    if done:
        print(f"  resuming; already done: {sorted(done)}")
    print(f"{'arm':>4} {'kb_pt':>7} {'kb_samp':>8} {'ratio':>6} {'Phi_pt':>7} "
          f"{'Phi_samp':>9} {'floor':>7} {'%pt':>6} {'%samp':>7} {'verdict':>9}")
    worst = 0.0; nviol = 0
    for arm in ARMS:
        if arm in done:
            r = next(x for x in recs if x["arm"] == arm)
            if not r["holds"]:
                nviol += 1
            worst = max(worst, r["frac_sampler"])
            print(f"{arm:>4} {r['kbar_pt']:7.3f} {r['kbar_sampler']:8.3f} "
                  f"{r['ratio']:6.3f} {r['phi_pt_s2']:7.3f} {r['phi_sampler_s2']:9.3f} "
                  f"{r['floor_meas_s2']:7.3f} {100*r['frac_pt']:5.1f}% "
                  f"{100*r['frac_sampler']:6.1f}% "
                  f"{'OK' if r['holds'] else 'VIOLATED':>9}   (cached)")
            continue
        kp, ka, fl = [], [], []
        for sd in SEEDS:
            net = train(arm, sd)
            kp.append(profile_pt(net))
            k, v = sample_and_kbar(net, seed=900 + sd)
            ka.append(k); fl.append(v / SIG2)
        kpm, kam, flm = float(np.mean(kp)), float(np.mean(ka)), float(np.mean(fl))
        php = integrate(1e-6, kpm, SIG) / SIG2
        phs = integrate(1e-6, kam, SIG) / SIG2
        ok = phs < flm
        if not ok:
            nviol += 1
        worst = max(worst, phs / flm)
        print(f"{arm:>4} {kpm:7.3f} {kam:8.3f} {kam/kpm:6.3f} {php:7.3f} {phs:9.3f} "
              f"{flm:7.3f} {100*php/flm:5.1f}% {100*phs/flm:6.1f}% "
              f"{'OK' if ok else 'VIOLATED':>9}")
        rec = dict(key=f"COS_{arm}", arm=arm, kbar_pt=round(kpm, 4),
                   kbar_sampler=round(kam, 4), ratio=round(kam / kpm, 4),
                   phi_pt_s2=round(php, 4), phi_sampler_s2=round(phs, 4),
                   floor_meas_s2=round(flm, 4),
                   frac_pt=round(php / flm, 4), frac_sampler=round(phs / flm, 4),
                   holds=bool(ok), seeds=len(SEEDS))
        recs.append(rec)
        with open(OUT, "a") as f:          # append now, not at the end
            f.write(json.dumps(rec) + "\n")
    print()
    print(f"  arms violating their own bound: {nviol}/5")
    print(f"  worst arm sits at {100*worst:.1f}% of its measured floor")
    print(f"  ratio range across arms: "
          f"{min(r['ratio'] for r in recs):.3f} to {max(r['ratio'] for r in recs):.3f}"
          f"   (standard-protocol ratio was 0.83)")
    print(f"\nwrote {OUT}  ({time.time()-t_start:.0f}s)")
