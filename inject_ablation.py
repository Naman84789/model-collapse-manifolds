"""Break-test of Theorem I(a): is Phi_det really the DETERMINISTIC DEFICIT floor,
with the 2x excess in the measured recursion floor (9.84 sigma^2) being the stochastic
INJECTION channel?

Identical self-consuming recursion to head-to-head Arm A but at a FIXED t0=1e-4 (no
anneal, so the map is autonomous and lands on its own fixed point), no matcher, two arms:
  NOISE   : full reverse-SDE (drift + sqrt(dt) injection)  -> expect ~9 sigma^2 (= Arm A)
  NONOISE : same drift, injection term ZEROED (deterministic Euler) -> expect ~Phi_det
Theory (already computed, zero fit): Phi_det fixed point at t0=1e-4, kbar=3.9 = 4.54 sigma^2.

PASS = NONOISE lands on ~4.5 sigma^2 (deficit) and NOISE on ~9 sigma^2, so
       injection = NOISE - NONOISE ~ 5 sigma^2, i.e. the two channels are separated
       and the deterministic floor is a genuine LOWER bound. Resumable via jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"C:\Users\naman\Downloads\metric-audit\inject_ablation.jsonl"
R, SIGMA, T_MAX = 2.5, 0.05, 8.0
NSAMP, STEPS, BATCH, G, LAM, T0 = 4000, 1500, 512, 15, 0.5, 1e-4
SEEDS = [0, 1, 2]

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
def el(): return f"{(time.time()-t0_)/60:.1f} min"

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

def ring(n, rng, sig=SIGMA):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")

def train(data, seed):
    torch.manual_seed(seed); net = Score()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    dt_ = torch.tensor(data); n = len(dt_)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,)); x0 = dt_[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def sample(net, n, t0, seed, noise, k=200):
    torch.manual_seed(seed); x = torch.randn(n, 2); ts = np.geomspace(T_MAX, t0, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt
        if noise:
            x = x + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x.numpy()

def rvar(x): return float(np.linalg.norm(x, axis=1).var())

def recursion(seed, noise):
    rng = np.random.default_rng(seed)
    cur = ring(NSAMP, rng); traj = []
    for g in range(G):
        nreal = int(round(LAM * NSAMP)); real = ring(nreal, rng)
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        net = train(pool, seed * 100 + g)
        cur = sample(net, NSAMP, T0, seed * 7 + g, noise)
        traj.append(round(rvar(cur), 6))
    return traj

for noise in [True, False]:
    name = "noise" if noise else "nonoise"
    for seed in SEEDS:
        key = f"{name}_s{seed}"
        if key in done:
            print(f"{key} skip", flush=True); continue
        traj = recursion(seed, noise)
        log(dict(key=key, noise=noise, seed=seed, traj=traj))
        print(f"{key}: g0={traj[0]:.5f} gEND={traj[-1]:.5f} = {traj[-1]/SIGMA**2:.2f} s2  ({el()})", flush=True)

print(f"ALL DONE {el()}  (sigma^2={SIGMA**2}, Phi_det@t0=1e-4={0.01135}={0.01135/SIGMA**2:.2f}s2)", flush=True)
