"""
REPLAY-BUFFER BASELINE -- the missing control for the paper's 'why this is not a
replay buffer' paragraph, which until now was argued theoretically but never run.

Same harness as head_to_head.py arm A (std reverse-SDE, t0 annealed 0.05 -> 1e-4
geometrically, NO matcher), plus a replay arm: the SAME m=2000 stale real samples
that the anchor uses as its reference are instead INSERTED INTO the training pool
every generation (identical information budget to the anchor; the buffer is drawn
once at g=0 and never refreshed, matching the anchor's staleness).

Pool per generation = 2000 fresh real + 2000 stale replay + 2000 synthetic = 6000,
so the effective real fraction is lambda_eff = 2/3 and synthetic is a 1/3 MINORITY
(a harder test for the paper's claim than the 'synthetic majority' wording).

PREDICTION (Theorem law, affine closure, c ~ 0.05, e0^2 ~ 3.8 sigma^2):
  v_inf ~ ((1+c) * lambda_eff * sigma^2 + e0^2) / (lambda_eff (1+c) - c) ~ 6-7 sigma^2
i.e. replay should DILUTE the floor (arm A: 9.8 sigma^2 -> ~7 sigma^2) but NOT break
it; the anchor holds 1.01 sigma^2 with the same 2000 real points. If instead replay
also reaches ~1 sigma^2, the paper's replay-vs-anchor distinction is falsified.

5 seeds, G=20, resumable via replay_baseline.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
OUT = r"C:\Users\naman\Downloads\metric-audit\replay_baseline.jsonl"
R, SIGMA, T_MAX = 2.5, 0.05, 8.0
NSAMP, STEPS, BATCH, G, LAM = 4000, 2000, 512, 20, 0.5
SEEDS = [0, 1, 2, 3, 4]
M_REPLAY = 2000

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

def anneal_t0(g):
    return float(np.geomspace(0.05, 1e-4, G)[g])

def recursion(seed):
    rng = np.random.default_rng(seed)
    stale = ring(M_REPLAY, rng)          # same draw call order as head_to_head.py
    cur = ring(NSAMP, rng); traj = []
    for g in range(G):
        nreal = int(round(LAM * NSAMP)); real = ring(nreal, rng)
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        # replay: the stale reference goes INTO the pool (never refreshed)
        pool = np.concatenate([real, stale, cur[si]], 0).astype("float32")
        net = train(pool, seed * 100 + g)
        cur = std_sample(net, NSAMP, anneal_t0(g), seed * 7 + g)
        traj.append(round(rvar(cur), 6))
        print(f"  R_s{seed} g{g}: v={traj[-1]/SIGMA**2:.2f} sig^2  ({el()})", flush=True)
    return traj

for seed in SEEDS:
    key = f"R_replay_s{seed}"
    if key in done:
        print(f"{key} skip", flush=True); continue
    traj = recursion(seed)
    log(dict(key=key, arm="R", seed=seed, traj=traj))
    print(f"{key}: g0={traj[0]:.5f} g{G-1}={traj[-1]:.5f}  ({el()})", flush=True)

print(f"ALL DONE {el()}  (sigma^2={SIGMA**2})", flush=True)
