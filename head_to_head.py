"""
Items 2+3 (and empirical check of Item 1) — the headline figure.

Cambridge's method (annealed truncation, NO matcher) vs our fix, identical ring, G=20,
lambda=0.5, 5 seeds each. Three arms:
  A. ANNEAL  : std reverse-SDE, t0(g) annealed 0.05 -> 1e-4 geometrically, NO matcher.
               = 2606.13796's remedy. Theory: saturates at the recursion fixed point of
               the deterministic deficit floor Phi_det (Theorem I(a)), ~4-5 sigma^2.
  B. STD+MATCH: same sampler + bidirectional local matcher vs a stale reference. -> sigma^2.
  C. JUMP+MATCH: Tweedie jump (t_hat=0.05) + matcher. -> sigma^2, smaller tails.

Overlays the THEORY prediction: the deterministic recursion fixed point v* solving
v = Phi_det(lambda*sigma^2 + (1-lambda) v), Phi_det from deficit_floor_law integration.
If arm A saturates at v*, Theorem I(a) is empirically confirmed on the recursion.

5 seeds -> mean +/- std per generation (Item 3). Resumable via head_to_head.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"C:\Users\naman\Downloads\metric-audit\head_to_head.jsonl"
R, SIGMA, T_MAX = 2.5, 0.05, 8.0
NSAMP, STEPS, BATCH, G, LAM = 4000, 2000, 512, 20, 0.5
SEEDS = [0, 1, 2, 3, 4]
THAT = 0.05

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

def jump_sample(net, n, t_hat, seed):
    torch.manual_seed(seed); x = reverse_to(net, torch.randn(n, 2), t_hat)
    s = np.sqrt(1 - np.exp(-t_hat)); a = np.exp(-t_hat / 2)
    with torch.no_grad():
        e = net(x, torch.full((n, 1), float(t_hat)))
    return ((x - s * e) / a).numpy()

# bidirectional local (k=1 normal) matcher vs reference; ring => radial normal
def match_bidir(pts, ref, rng):
    rr = np.linalg.norm(ref, axis=1); v_ref = float(((rr - rr.mean()) ** 2).mean())
    rp = np.linalg.norm(pts, axis=1); mu = rp.mean(); h = rp - mu
    v_p = float((h ** 2).mean())
    if v_p > v_ref:
        hn = h * np.sqrt(v_ref / max(v_p, 1e-12))
    else:
        extra = rng.normal(size=len(pts)) * np.sqrt(max(v_ref - v_p, 0.0))
        hn = h + extra
    rn = mu + hn; u = pts / np.maximum(rp[:, None], 1e-9)
    return (u * rn[:, None]).astype("float32")

def rvar(x):
    return float(np.linalg.norm(x, axis=1).var())

def anneal_t0(g):
    return float(np.geomspace(0.05, 1e-4, G)[g])

def recursion(seed, arm):
    rng = np.random.default_rng(seed)
    stale = ring(2000, rng)
    cur = ring(NSAMP, rng); traj = []
    for g in range(G):
        nreal = int(round(LAM * NSAMP)); real = ring(nreal, rng)
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        if arm in ("B", "C"):
            pool = match_bidir(pool, stale, rng)
        net = train(pool, seed * 100 + g)
        if arm == "A":
            cur = std_sample(net, NSAMP, anneal_t0(g), seed * 7 + g)
        elif arm == "B":
            cur = std_sample(net, NSAMP, 0.02, seed * 7 + g)
            cur = match_bidir(cur, stale, rng)
        else:
            cur = jump_sample(net, NSAMP, THAT, seed * 7 + g)
            cur = match_bidir(cur, stale, rng)
        traj.append(round(rvar(cur), 6))
    return traj

ARMS = {"A": "anneal_nomatch", "B": "std_match", "C": "jump_match"}
for arm, name in ARMS.items():
    for seed in SEEDS:
        key = f"{arm}_{name}_s{seed}"
        if key in done:
            print(f"{key} skip", flush=True); continue
        traj = recursion(seed, arm)
        log(dict(key=key, arm=arm, seed=seed, traj=traj))
        print(f"{key}: g0={traj[0]:.5f} g{G-1}={traj[-1]:.5f}  ({el()})", flush=True)

print(f"ALL DONE {el()}  (sigma^2={SIGMA**2})", flush=True)
