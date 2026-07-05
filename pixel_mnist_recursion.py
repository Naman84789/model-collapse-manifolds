"""
Item 4 — a REAL pixel-space generative recursion (not latents, not synthetic manifolds).

MNIST downsampled to 8x8 = 64-dim PIXEL space (a genuine ~10-14 dim data manifold in
64-dim ambient). Small diffusion MLP trained from scratch each generation. Self-consuming
recursion, lambda=0.5, G=8, 3 seeds. Two arms:
  UNFIXED : train on pool, sample (std reverse-SDE t0=0.02), recurse -> expect collapse.
  FIXED   : + bidirectional LOCAL (kNN-PCA, kdim=12) matcher vs a stale real reference.
Collapse metric: off-manifold distance = mean nearest-neighbor L2 from generated samples
to a held-out REAL MNIST test set (pixel space). True-data baseline included.

CPU-only friendly: 64-dim, 256-wide MLP, 1500 steps (~0.4 min/gen). ~30-40 min total.
Resumable via pixel_mnist_recursion.jsonl. Needs torchvision OR a cached mnist8x8.npz.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"C:\Users\naman\Downloads\metric-audit\pixel_mnist_recursion.jsonl"
CACHE = r"C:\Users\naman\Downloads\metric-audit\mnist8x8.npz"
T_MAX, STEPS, BATCH, G, LAM, NSAMP = 8.0, 1500, 256, 8, 0.5, 3000
D = 64
SEEDS = [0, 1, 2]

def load_mnist8():
    if os.path.exists(CACHE):
        z = np.load(CACHE); return z["train"], z["test"]
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    def grab(train):
        ds = datasets.MNIST(root=r"C:\Users\naman\Downloads\metric-audit\_mnist",
                            train=train, download=True)
        X = ds.data.float() / 255.0                      # (N,28,28)
        X = F.adaptive_avg_pool2d(X.unsqueeze(1), (8, 8)).squeeze(1)  # (N,8,8)
        return (X.reshape(len(X), 64).numpy() * 2 - 1).astype("float32")  # to [-1,1]
    tr, te = grab(True), grab(False)
    np.savez(CACHE, train=tr, test=te); return tr, te

TRAIN, TEST = load_mnist8()
TESTREF = TEST[:4000]

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
    def __init__(self, Dd=D, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(Dd + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, Dd))
    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))

def train(data, seed):
    torch.manual_seed(seed); net = Score()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    dt_ = torch.tensor(data); n = len(dt_)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,)); x0 = dt_[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, D); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def std_sample(net, n, t0, seed, k=200):
    torch.manual_seed(seed); x = torch.randn(n, D); ts = np.geomspace(T_MAX, t0, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x.numpy().astype("float32")

def knn_idx(pts, ref, k):
    out = np.empty((len(pts), k), dtype=np.int64)
    for i in range(0, len(pts), 200):
        c = pts[i:i + 200]
        d2 = ((c[:, None, :] - ref[None, :, :]) ** 2).sum(2)
        out[i:i + 200] = np.argpartition(d2, k, axis=1)[:, :k]
    return out

def axmatch_bidir(pts, ref, k=96, kdim=12, rng=None):
    """Local per-axis bidirectional matcher (grand_harness MN style), pixel space.

    For each pool point, build a local chart from its k nearest REFERENCE points:
    top-kdim PCA dirs = tangent, remaining = normal. The reference's own local normal
    variance is exactly the small eigenvalues of that neighborhood covariance. Project
    the point onto the tangent and rescale its normal coordinates to that variance --
    pulls gross outliers in (deficit) and re-inflates thinned axes (injection).
    k > D=64 keeps the DxD covariance full-rank."""
    idx_p = knn_idx(pts, ref, k)
    out = pts.copy()
    for i in range(len(pts)):
        nb = ref[idx_p[i]]; mu = nb.mean(0)
        w, V = np.linalg.eigh(np.cov((nb - mu).T))   # eigenvalues ascending
        T = V[:, -kdim:]                             # tangent dirs (top var)
        Nrm = V[:, :-kdim]                           # normal dirs (low var)
        vr = np.clip(w[:-kdim], 0.0, None)           # local reference normal variance
        rel = pts[i] - mu; coeff = rel @ Nrm
        sc = np.sqrt(vr / (coeff ** 2 + 1e-9))
        out[i] = mu + T @ (T.T @ rel) + Nrm @ (coeff * sc)
    return out.astype("float32")

def offman(X, ref):
    dmin = np.empty(len(X), dtype="float32")
    for i in range(0, len(X), 200):
        c = X[i:i + 200]
        d2 = ((c[:, None, :] - ref[None, :, :]) ** 2).sum(2)
        dmin[i:i + 200] = np.sqrt(d2.min(1))
    return float(np.mean(dmin))

TRUE = offman(TEST[4000:8000], TESTREF)
print(f"true-data off-manifold baseline = {TRUE:.4f}  ({el()})", flush=True)
log(dict(key="TRUE_baseline", offman=round(TRUE, 4)))

def recursion(seed, fixed):
    rng = np.random.default_rng(seed)
    stale = TRAIN[rng.choice(len(TRAIN), 2000, replace=False)]
    cur = TRAIN[rng.choice(len(TRAIN), NSAMP, replace=False)].copy(); traj = []
    for g in range(G):
        nreal = int(round(LAM * NSAMP))
        real = TRAIN[rng.choice(len(TRAIN), nreal, replace=False)]
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        if fixed:
            pool = axmatch_bidir(pool, stale, rng=rng)
        net = train(pool, seed * 100 + g)
        cur = std_sample(net, NSAMP, 0.02, seed * 7 + g)
        if fixed:
            cur = axmatch_bidir(cur, stale, rng=rng)
        traj.append(round(offman(cur, TESTREF), 4))
        print(f"  seed{seed} {'FIX' if fixed else 'UNF'} g{g}: offman={traj[-1]:.4f}  ({el()})", flush=True)
    return traj

for fixed in [False, True]:
    name = "fixed" if fixed else "unfixed"
    for seed in SEEDS:
        key = f"{name}_s{seed}"
        if key in done:
            print(f"{key} skip", flush=True); continue
        traj = recursion(seed, fixed)
        log(dict(key=key, fixed=fixed, seed=seed, traj=traj))
        print(f"{key}: g0={traj[0]:.4f} g{G-1}={traj[-1]:.4f}  ({el()})", flush=True)

print(f"ALL DONE {el()}  (true={TRUE:.4f})", flush=True)
