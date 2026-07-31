"""
MNIST kbar(w) -- real-data test of the capacity-degradation law.

The paper measures the slope-ceiling degradation kbar(w) on synthetic tubes only
(ring / segment / sphere / R^10 circle). This closes the named limitation "real-data
kbar(w) not measured" by re-running the SAME extraction convention on genuine MNIST 8x8
pixel data (64-dim), whose digit manifold has a real thin-normal structure (measured
intrinsic normal std sig0 ~ 0.072, tangent std ~0.41, ratio ~0.17).

Protocol (mirrors capacity_domains.py):
  - Local normal frames from clean digits via kNN-PCA (k=96, kdim=12): top-12 PCA dirs
    = tangent, remaining 52 = normal. Frames come from the CLEAN manifold.
  - Fatten NORMAL-ONLY to width sig: x0 = m + sig * (Nrm @ eta), eta~N(0,I_52), so the
    added per-axis normal 2nd moment is sig^2 and the tangent is untouched.
  - Train a fresh score MLP (3x256 SiLU, 2000 steps) on the fattened tube.
  - Extract kbar: regress the network's normal-projected eps on the normal coordinate of
    x_t at t in {0.1,0.05,0.02,0.01,0.005}, pooled over all 52 normal axes; kbar = max_t.
  - Total normal width w = sig0^2 + sig^2 (per-axis); fit kbar = A - B*sqrt(w).

PREDICTION: kbar falls with sqrt(w) at a comparable relative rate to the tubes (~25-50%),
with an MNIST-specific constant. FALSIFIER: flat kbar(w) -> the degradation is a
synthetic-tube artifact and the limitation stands. Reported honestly either way.

2 net seeds per width. Resumable via mnist_kbar_w.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
D = 64
OUT = r"C:\Users\naman\Downloads\metric-audit\mnist_kbar_w.jsonl"
CACHE = r"C:\Users\naman\Downloads\metric-audit\mnist8x8.npz"
T_MAX, STEPS, BATCH = 8.0, 2000, 512
K, KDIM = 96, 12
NREF, NBASE, NMEAS = 8000, 6000, 1200
SIGS = [0.0, 0.05, 0.10, 0.15, 0.22]
TS_FIT = [0.1, 0.05, 0.02, 0.01, 0.005]
SEEDS = [500, 501]

z = np.load(CACHE); TRAIN = z["train"]
t0_ = time.time()
def el(): return f"{(time.time()-t0_)/60:.1f} min"

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done.add(json.loads(line)["key"])
        except Exception: pass
def log(rec):
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")

def knn_idx(pts, ref, k):
    out = np.empty((len(pts), k), np.int64)
    for i in range(0, len(pts), 200):
        c = pts[i:i + 200]; d2 = ((c[:, None] - ref[None]) ** 2).sum(2)
        out[i:i + 200] = np.argpartition(d2, k, 1)[:, :k]
    return out

def frames(base, ref):
    """Per-point normal basis Nrm (n,64,52) and mean small-eig (intrinsic normal var)."""
    idx = knn_idx(base, ref, K)
    Nrm = np.empty((len(base), D, D - KDIM), np.float32)
    small = np.empty(len(base), np.float32)
    for i in range(len(base)):
        nb = ref[idx[i]]; w, V = np.linalg.eigh(np.cov((nb - nb.mean(0)).T))
        Nrm[i] = V[:, :-KDIM]                # normal dirs = low-variance eigenvectors
        small[i] = w[:-KDIM].mean()
    return Nrm, float(small.mean())

def temb(t, dim=32):
    f = torch.exp(torch.linspace(0, 5, dim // 2)).to(t)
    return torch.cat([torch.sin(t * f), torch.cos(t * f)], 1)

class Score(nn.Module):
    def __init__(self, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))
    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))

def train(data, seed):
    torch.manual_seed(seed); net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    dt = torch.tensor(data); n = len(dt)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,)); x0 = dt[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        zz = torch.randn(BATCH, D); xt = a * x0 + s * zz
        loss = ((net(xt, t) - zz) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def slope_at(net, m, Nrm, off, sig, t):
    """Regress normal-projected eps on the normal coord of x_t, pooled over 52 axes."""
    a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
    rng = np.random.default_rng(int(t * 1e4) + 7)
    x0 = m + sig * off                       # fattened tube point (normal-only)
    zz = rng.normal(size=(len(m), D)).astype("float32")
    xt = (a * x0 + s * zz).astype("float32")
    with torch.no_grad():
        e = net(torch.tensor(xt), torch.full((len(m), 1), float(t))).numpy()
    c = np.einsum("nde,nd->ne", Nrm, xt - a * m)      # normal coord of x_t  (n,52)
    ec = np.einsum("nde,nd->ne", Nrm, e)              # eps normal comp      (n,52)
    c = c.ravel(); ec = ec.ravel()
    return float(np.cov(c, ec)[0, 1] / max(c.var(), 1e-12))

# ---- frames (clean manifold) ----
rng = np.random.default_rng(0)
REF = TRAIN[rng.choice(len(TRAIN), NREF, replace=False)]
BASE = TRAIN[rng.choice(len(TRAIN), NBASE, replace=False)]
MEAS = TRAIN[rng.choice(len(TRAIN), NMEAS, replace=False)]
print("computing frames ...", flush=True)
Nrm_b, sig0v = frames(BASE, REF)
Nrm_m, _ = frames(MEAS, REF)
sig0 = float(np.sqrt(sig0v))
# fixed per-point normal offsets (unit-eta), scaled by sig at use
rb = np.random.default_rng(1); rm = np.random.default_rng(2)
off_b = np.einsum("nde,ne->nd", Nrm_b, rb.normal(size=(NBASE, D - KDIM)).astype("float32"))
off_m = np.einsum("nde,ne->nd", Nrm_m, rm.normal(size=(NMEAS, D - KDIM)).astype("float32"))
print(f"intrinsic normal sig0 = {sig0:.4f} (var {sig0v:.5f})   ({el()})", flush=True)
log(dict(key="MNIST_sig0", sig0=round(sig0, 4), sig0_var=round(sig0v, 5)))

print(f"\n{'sig':>5} {'sqrt(w)':>8} {'seed':>5} {'kbar':>6}", flush=True)
res = {}
for sig in SIGS:
    wtot = sig0v + sig ** 2                   # total per-axis normal 2nd moment
    for seed in SEEDS:
        key = f"MK_sig{sig}_s{seed}"
        if key in done:
            for line in open(OUT):
                d = json.loads(line)
                if d.get("key") == key:
                    res.setdefault(sig, []).append((np.sqrt(wtot), d["kbar"]))
            print(f"{key} skip", flush=True); continue
        net = train((BASE + sig * off_b).astype("float32"), seed)
        ks = [slope_at(net, MEAS, Nrm_m, off_m, sig, t) for t in TS_FIT]
        kb = float(max(ks))
        log(dict(key=key, sig=sig, sqrt_w=round(float(np.sqrt(wtot)), 4),
                 kbar=round(kb, 3), k_by_t=[round(k, 3) for k in ks], seed=seed))
        res.setdefault(sig, []).append((np.sqrt(wtot), kb))
        print(f"{sig:>5} {np.sqrt(wtot):>8.4f} {seed:>5} {kb:>6.2f}   ({el()})", flush=True)

# ---- fit kbar = A - B sqrt(w) over seed-averaged points ----
pts = sorted((sw, np.mean([k for s, k in v if abs(s - sw) < 1e-9]))
             for sw, v in [(np.mean([p[0] for p in vv]), vv) for _, vv in res.items()])
sqw = np.array([p[0] for p in pts]); kb = np.array([p[1] for p in pts])
B, A = np.polyfit(sqw, kb, 1)
resid = kb - (A + B * sqw)
rel = (kb[0] - kb[-1]) / kb[0] * 100
print(f"\nMNIST kbar(w) = {A:.2f} {B:+.1f} sqrt(w)   max|res|={np.abs(resid).max():.3f}", flush=True)
print(f"  degradation {kb[0]:.2f} -> {kb[-1]:.2f}   ({rel:.0f}% over sqrt(w) {sqw[0]:.3f}->{sqw[-1]:.3f})", flush=True)
print(f"  [tube references: ring 24%, segment 42%, sphere 30%, R^10 49%]", flush=True)
log(dict(key="LAW_MNIST", A=round(float(A), 3), B=round(float(B), 3),
         maxres=round(float(np.abs(resid).max()), 3), rel_drop_pct=round(float(rel), 1),
         sqrt_w=[round(float(x), 4) for x in sqw], kbar=[round(float(x), 3) for x in kb]))
print(f"ALL DONE {el()}", flush=True)
