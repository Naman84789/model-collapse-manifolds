"""
CEILING ORIGIN -- what actually sets kbar ~ 3.9?  (the fundamental question under the
1.5-2.0x residual: the closure treats kbar as given; nothing in the paper says WHY the
optimizer stalls at 40% of the demanded peak slope.)

Three candidate mechanisms, one arm each, sigma = 0.05 ring (kappa_max = 10):

  A BASE   : standard protocol (control; expect kbar ~ 3.9-4.4).
  B TMEAS  : training-measure fix. 50% of each batch draws t ~ LogUniform(2e-4, 0.1)
             (the deficit band and below), 50% standard U(1e-3, 8).  The band holds
             ~0.8% of the standard measure and t < 1e-3 is NEVER trained, though the
             sampler runs to 1e-4.  If the ceiling is gradient starvation, B lifts it.
  C EMB    : embedding-resolution fix. Frequencies exp(linspace(0, 8, 16)) instead of
             exp(linspace(0, 5, 16)): f_max 148 -> 2981, expressible t-resolution
             ~0.007 -> ~3e-4 (the band lives below 0.007).  If the ceiling is
             expressivity in t, C lifts it.
  D BOTH   : B + C.
  E STEPS4 : 4x training steps (8000), standard measure/embedding.  If the ceiling is
             just unconverged SGD, E lifts it.

For every arm: fine khat(t) profile (regression under p_t on the sampler grid),
kbar = max, AND the measured t0=1e-4 sampling floor v_full (3 sampler seeds).
PAYOFF: the floor law g(rho), rho = 2 sigma kbar, must predict each arm's measured
floor from its own kbar with zero free parameters -- a second, out-of-protocol
zero-fit test.  E.g. kbar 3.9 -> 3.8 sigma^2 but kbar 7 -> 1.4 sigma^2.
Resumable via ceiling_origin.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
OUT = r"C:\Users\naman\Downloads\metric-audit\ceiling_origin.jsonl"
R, SIG, T_MAX, BATCH, N = 2.5, 0.05, 8.0, 512, 6000
T0, KGRID = 1e-4, 200
TS = np.geomspace(T_MAX, T0, KGRID + 1)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            pass

def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")

t0_ = time.time()
def el():
    return f"{(time.time()-t0_)/60:.1f} min"

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

def profile_pt(net, n=4000, seed=7):
    rng = np.random.default_rng(seed); ks = []
    for t in TS[:-1]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        th = rng.uniform(0, 2 * np.pi, n)
        rr = a * (R + SIG * rng.normal(size=n))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(n, 2))
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((n, 1), float(t))).numpy() * uh).sum(1)
        h = rad - rad.mean()
        ks.append(float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12)))
    return np.array(ks)

def std_sample_var(net, seed, n=4000):
    torch.manual_seed(seed); x = torch.randn(n, 2)
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return float(np.linalg.norm(x.numpy(), axis=1).var())

def g_floor(rho):
    """no-overshoot closed form g(rho) of Theorem 2 (valid rho < 1)."""
    if rho >= 1:
        return 1.0
    q = np.sqrt(1 - rho * rho)
    um, up = ((1 - q) / rho) ** 2, ((1 + q) / rho) ** 2
    wexit = (1 + up) * np.exp(-4 * q) + ((3 - 2 * q) - (3 + 2 * q) * np.exp(-4 * q)) / (2 * rho * rho)
    return 1 + (wexit - (1 + um)) / (1 + um) ** 2

ARMS = ["A", "B", "C", "D", "E"]
u = SIG * SIG
print("arm  kbar   floor_pred(law)   floor_meas   (sigma^2 units)", flush=True)
for arm in ARMS:
    key = f"CO_{arm}"
    if key in done:
        print(f"{key} skip", flush=True); continue
    net = train(arm)
    ks = profile_pt(net)
    kbar = float(ks.max())
    rho = 2 * SIG * kbar
    pred = g_floor(rho)
    vs = [std_sample_var(net, 900 + s) for s in (0, 1, 2)]
    vm = float(np.mean(vs))
    log(dict(key=key, arm=arm, kbar=round(kbar, 3), rho=round(rho, 3),
             floor_pred_sig2=round(pred, 3), floor_meas_sig2=round(vm / u, 3),
             v_seeds=[round(v, 6) for v in vs],
             khat=[round(float(k), 4) for k in ks],
             ts=[round(float(t), 6) for t in TS[:-1]]))
    print(f"{arm:>3} {kbar:>5.2f}   {pred:>8.2f}          {vm/u:>7.2f}       ({el()})",
          flush=True)
print(f"ALL DONE {el()}", flush=True)
