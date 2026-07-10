"""
SUBCRITICAL-LAMBDA RUN -- exhibits the critical-fraction regime of Theorem law
part (ii), which every previous experiment (lambda >= 0.25) left untested.

Protocol matches the two-horned ablation: fixed t0 = 1e-4, stochastic reverse-SDE
sampler, NO anchor. Two arms:

  L02 : lambda = 0.02, G = 25.  The measured saturated slope c_inf ~ 0.03-0.05
        (poolwidth probe fat end; rate-based estimate ~0) gives
        lambda* = c_inf/(1+c_inf) ~ 0.03-0.05, so lambda=0.02 is subcritical or
        near-critical.  PREDICTION: no plateau within the horizon; v climbs
        roughly +e0^2 ~ +3.8 sigma^2 per generation (slowly compounding), through
        ~50-100 sigma^2 by g=25.  (A finite horizon cannot distinguish true
        divergence from a plateau >~300 sigma^2; the claim is the CONTRAST with
        the supercritical arm, which flattens early.)

  L25 : lambda = 0.25, G = 15.  PREDICTION: plateau ~18 sigma^2 by ~g=10
        (affine law with c~0.03, e0^2~3.8 sigma^2; the measured capacity
        degradation on fat pools may push the plateau somewhat higher).

3 seeds each; resumable via subcritical_lambda.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
OUT = r"C:\Users\naman\Downloads\metric-audit\subcritical_lambda.jsonl"
R, SIGMA, T_MAX = 2.5, 0.05, 8.0
NSAMP, STEPS, BATCH = 4000, 2000, 512
T0 = 1e-4
SEEDS = [0, 1, 2]
ARMS = {"L02": (0.02, 25), "L25": (0.25, 15)}

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

def reverse_to(net, x, t0, k=200):
    n = len(x); ts = np.geomspace(T_MAX, t0, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x

def std_sample(net, n, t0, seed):
    torch.manual_seed(seed); return reverse_to(net, torch.randn(n, 2), t0).numpy()

def rvar(x):
    return float(np.linalg.norm(x, axis=1).var())

def recursion(seed, lam, G, tag):
    rng = np.random.default_rng(seed)
    cur = ring(NSAMP, rng); traj = []
    for g in range(G):
        nreal = int(round(lam * NSAMP)); real = ring(nreal, rng)
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        net = train(pool, seed * 100 + g)
        cur = std_sample(net, NSAMP, T0, seed * 7 + g)
        traj.append(round(rvar(cur), 6))
        print(f"  {tag}_s{seed} g{g}: v={traj[-1]/SIGMA**2:.2f} sig^2  ({el()})",
              flush=True)
    return traj

for tag, (lam, G) in ARMS.items():
    for seed in SEEDS:
        key = f"{tag}_s{seed}"
        if key in done:
            print(f"{key} skip", flush=True); continue
        traj = recursion(seed, lam, G, tag)
        log(dict(key=key, arm=tag, lam=lam, seed=seed, traj=traj))
        print(f"{key}: g0={traj[0]:.5f} g{G-1}={traj[-1]:.5f}  ({el()})", flush=True)

print(f"ALL DONE {el()}  (sigma^2={SIGMA**2})", flush=True)
