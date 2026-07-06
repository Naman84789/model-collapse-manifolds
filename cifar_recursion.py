"""
CIFAR-10 pixel-space self-consuming recursion (GPU) -- the scale step reviewers asked for.

Same protocol as pixel_mnist_recursion.py, lifted to 32x32x3 = 3072-dim pixel space:
  UNFIXED : train small UNet on pool, sample (std reverse-SDE t0=0.02), recurse.
  FIXED   : + bidirectional LOCAL matcher vs a stale real reference (m=2000, drawn once).
Collapse metric: off-manifold distance = mean nearest-neighbor L2 from generated samples
to a held-out REAL CIFAR test set (pixel space). True-data baseline included.

Matcher generalization for D >> m: local kNN chart from the reference (Gram-trick SVD,
rank k-1 < D). Per-axis bidirectional match on the resolvable normal spectrum
(axes kdim..J); scalar energy match on the unresolved orthogonal remainder, whose
per-axis reference variance is estimated from the reference's own off-chart residuals.
This is the paper's operator with 'per-axis where measurable' honesty.

Resumable at GENERATION granularity: cifar_recursion.jsonl + per-arm state .npz.
~18-25 min/generation on an RTX 3050 laptop (AMP). G=8, 2 arms => ~5-6 h per seed.
Sample grids saved to cifar_grids/ each generation for qualitative inspection.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(12)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "cifar_recursion.jsonl")
STATE_DIR = os.path.join(BASE, "cifar_state"); os.makedirs(STATE_DIR, exist_ok=True)
GRID_DIR = os.path.join(BASE, "cifar_grids"); os.makedirs(GRID_DIR, exist_ok=True)

T_MAX, STEPS, BATCH, G, LAM, NSAMP = 8.0, 6000, 128, 8, 0.5, 6000
D = 3072
SEEDS = [0, 1, 2]
K_NN, KDIM, J_AXES = 160, 24, 128   # matcher: neighbors, tangent dim, resolvable axes

# ---------------------------------------------------------------- data
def _decode_parquet_cifar(path):
    """CIFAR-10 parquet (HF uoft-cs/cifar10): column 'img' holds PNG-encoded bytes
    per row (as {'bytes': ...} structs), 'label' the class. Returns (N,3072) float32
    in [-1,1], channel-major (CHW-flat) to match the rest of the pipeline."""
    import io, pyarrow.parquet as pq
    from PIL import Image
    tbl = pq.read_table(path)
    col = tbl.column("img").to_pylist()
    out = np.empty((len(col), D), dtype=np.float32)
    for i, cell in enumerate(col):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)  # (32,32,3) HWC
        out[i] = (im.transpose(2, 0, 1).reshape(D) / 255.0 * 2 - 1)                    # CHW-flat, [-1,1]
    return out

def load_cifar():
    """Loads CIFAR-10 from HuggingFace parquet (uoft-cs/cifar10), decoded locally.

    We avoid torchvision's downloader on purpose: its redirect chain currently hits
    a host with an EXPIRED SSL certificate, and the canonical Toronto host throttles
    to ~9 KB/s. The HF CDN is fast and reliable. Parquet files are fetched once via
    curl (see the download step) into _cifar_parquet/; this function decodes + caches
    to cifar32.npz. Raises with instructions if the parquet isn't present."""
    cache = os.path.join(BASE, "cifar32.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["train"], z["test"]
    pdir = os.path.join(BASE, "_cifar_parquet")
    tr_p, te_p = os.path.join(pdir, "train.parquet"), os.path.join(pdir, "test.parquet")
    if not (os.path.exists(tr_p) and os.path.exists(te_p)):
        raise FileNotFoundError(
            f"Parquet not found in {pdir}. Fetch once (fast HF CDN):\n"
            "  for s in train test; do curl -sL --retry 5 -o "
            f"\"{pdir}/$s.parquet\" "
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
            "plain_text/$s-00000-of-00001.parquet; done")
    a, b = _decode_parquet_cifar(tr_p), _decode_parquet_cifar(te_p)
    np.savez(cache, train=a, test=b); return a, b

TRAIN, TEST = load_cifar()
TESTREF = TEST[:4000]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done.add(json.loads(line)["key"])
        except Exception: pass

def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")

t0_ = time.time()
def el(): return f"{(time.time()-t0_)/60:.1f} min"

# ---------------------------------------------------------------- small UNet
class RB(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin);  self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.tp = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.tp(te)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, ch=64, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2 = ch, ch * 2
        self.cin  = nn.Conv2d(3, c1, 3, padding=1)
        self.d0a, self.d0b = RB(c1, c1, tdim), RB(c1, c1, tdim)
        self.down0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)      # 32->16
        self.d1a, self.d1b = RB(c1, c2, tdim), RB(c2, c2, tdim)
        self.down1 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)      # 16->8
        self.m1, self.m2 = RB(c2, c2, tdim), RB(c2, c2, tdim)
        self.u1a, self.u1b = RB(c2 + c2, c2, tdim), RB(c2, c1, tdim)
        self.u0a, self.u0b = RB(c1 + c1, c1, tdim), RB(c1, c1, tdim)
        self.nout = nn.GroupNorm(8, c1); self.cout = nn.Conv2d(c1, 3, 3, padding=1)
    def temb(self, t):
        f = torch.exp(torch.linspace(0, 5, 64, device=t.device))
        e = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)
        return self.tmlp(e)
    def forward(self, x, t):
        te = self.temb(t)
        h0 = self.cin(x)
        h0 = self.d0b(self.d0a(h0, te), te)                 # 32, c1
        h1 = self.down0(h0)
        h1 = self.d1b(self.d1a(h1, te), te)                 # 16, c2
        h2 = self.down1(h1)
        h2 = self.m2(self.m1(h2, te), te)                   # 8, c2
        u1 = F.interpolate(h2, scale_factor=2, mode="nearest")
        u1 = self.u1b(self.u1a(torch.cat([u1, h1], 1), te), te)   # 16, c1
        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = self.u0b(self.u0a(torch.cat([u0, h0], 1), te), te)   # 32, c1
        return self.cout(F.silu(self.nout(u0)))

# ---------------------------------------------------------------- train / sample
def train_net(data_flat, seed):
    torch.manual_seed(seed)
    net = UNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-4)
    scaler = torch.amp.GradScaler(DEV) if DEV == "cuda" else None
    dt_ = torch.tensor(data_flat.reshape(-1, 3, 32, 32))
    n = len(dt_); tick = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,))
        x0 = dt_[idx].to(DEV, non_blocking=True)
        t = torch.rand(BATCH, device=DEV) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2)[:, None, None, None]
        s = torch.sqrt(1 - torch.exp(-t))[:, None, None, None]
        z = torch.randn_like(x0); xt = a * x0 + s * z
        if scaler:
            with torch.amp.autocast(DEV):
                loss = ((net(xt, t) - z) ** 2).mean()
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss = ((net(xt, t) - z) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if step == 199:
            print(f"    [{200/(time.time()-tick):.1f} it/s]", flush=True)
    return net

@torch.no_grad()
def std_sample(net, n, t0, seed, k=200, chunk=500):
    torch.manual_seed(seed)
    ts = np.geomspace(T_MAX, t0, k + 1)
    outs = []
    for c0 in range(0, n, chunk):
        m = min(chunk, n - c0)
        x = torch.randn(m, 3, 32, 32, device=DEV)
        for i in range(k):
            t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
            s = np.sqrt(1 - np.exp(-t))
            with torch.amp.autocast(DEV) if DEV == "cuda" else torch.no_grad():
                e = net(x, torch.full((m,), t, device=DEV)).float()
            x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
        outs.append(x.reshape(m, D).cpu().numpy().astype("float32"))
    return np.concatenate(outs, 0)

# ---------------------------------------------------------------- GPU kNN + metric
def knn_idx(pts, ref, k, chunk=256):
    """indices of k nearest ref points for each pts row (L2), GPU-chunked."""
    R = torch.tensor(ref, device=DEV)
    rn = (R * R).sum(1)
    out = np.empty((len(pts), k), dtype=np.int64)
    for i in range(0, len(pts), chunk):
        C = torch.tensor(pts[i:i + chunk], device=DEV)
        d2 = (C * C).sum(1)[:, None] + rn[None, :] - 2.0 * (C @ R.T)
        out[i:i + chunk] = torch.topk(d2, k, dim=1, largest=False).indices.cpu().numpy()
    return out

def offman(X, ref, chunk=256):
    R = torch.tensor(ref, device=DEV)
    rn = (R * R).sum(1)
    dmin = np.empty(len(X), dtype="float32")
    for i in range(0, len(X), chunk):
        C = torch.tensor(X[i:i + chunk], device=DEV)
        d2 = (C * C).sum(1)[:, None] + rn[None, :] - 2.0 * (C @ R.T)
        dmin[i:i + chunk] = torch.clamp(d2.min(1).values, min=0).sqrt().cpu().numpy()
    return float(np.mean(dmin))

# ---------------------------------------------------------------- matcher (D >> m)
def axmatch_bidir(pts, ref, k=K_NN, kdim=KDIM, jax=J_AXES):
    """Bidirectional local matcher for high ambient D.

    Local chart per point from its k nearest REFERENCE points via Gram-trick SVD
    (rank k-1). Axes 0..kdim-1 (top variance) = tangent: passed through. Axes
    kdim..jax-1 = resolvable normal spectrum: per-axis bidirectional match of the
    point's coordinate to the reference's local variance on that axis. The orthogonal
    remainder (unresolvable, D-jax dims) is matched as a single energy scalar against
    the reference's own mean off-chart residual energy."""
    idx_p = knn_idx(pts, ref, k)
    out = pts.copy()
    for i in range(len(pts)):
        nb = ref[idx_p[i]]                      # (k, D)
        mu = nb.mean(0)
        Xc = nb - mu                            # (k, D)
        Gm = (Xc @ Xc.T) / (k - 1)              # (k, k) Gram
        w, U = np.linalg.eigh(Gm)               # ascending
        w = np.clip(w, 0, None)
        selg = np.argsort(w)[::-1][:jax]        # top jax eigenpairs
        lam = w[selg]                           # local ref variances, descending
        nz = lam > 1e-12
        Vax = (Xc.T @ U[:, selg][:, nz]) / np.sqrt(lam[nz] * (k - 1) + 1e-30)  # (D, r)
        lam = lam[nz]
        r = Vax.shape[1]
        kd = min(kdim, r - 1)
        rel = pts[i] - mu
        coeff = rel @ Vax                       # coordinates in local axes
        # reference off-chart residual energy per point (for the remainder match)
        res_ref = np.mean(np.maximum(
            (Xc * Xc).sum(1) - ((Xc @ Vax) ** 2).sum(1), 0.0))
        # tangent: keep; resolvable normal: per-axis bidirectional
        newc = coeff.copy()
        sc = np.sqrt(lam[kd:] / (coeff[kd:] ** 2 + 1e-9))
        newc[kd:] = coeff[kd:] * sc
        # remainder: scalar energy match
        res_vec = rel - coeff @ Vax.T
        res_en = float((res_vec * res_vec).sum())
        gscale = np.sqrt(res_ref / (res_en + 1e-9))
        out[i] = mu + newc @ Vax.T + res_vec * gscale
    return out.astype("float32")

# ---------------------------------------------------------------- sample grids
def save_grid(X, path, n=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    im = X[:n].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    im = np.clip((im + 1) / 2, 0, 1)
    rows = int(np.sqrt(n))
    fig, axes = plt.subplots(rows, rows, figsize=(rows, rows))
    for a, img in zip(axes.flat, im):
        a.imshow(img); a.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=120); plt.close(fig)

# ---------------------------------------------------------------- recursion
TRUE_KEY = "TRUE_baseline"
if TRUE_KEY not in done:
    TRUE = offman(TEST[4000:8000], TESTREF)
    log(dict(key=TRUE_KEY, offman=round(TRUE, 4)))
else:
    TRUE = [json.loads(l)["offman"] for l in open(OUT)
            if json.loads(l)["key"] == TRUE_KEY][0]
print(f"device={DEV}  true-data baseline = {TRUE:.4f}  ({el()})", flush=True)

def recursion(seed, fixed):
    name = "fixed" if fixed else "unfixed"
    rng = np.random.default_rng(seed)
    stale = TRAIN[rng.choice(len(TRAIN), 2000, replace=False)]
    state_f = os.path.join(STATE_DIR, f"{name}_s{seed}.npz")
    start_g, cur = 0, None
    if os.path.exists(state_f):
        z = np.load(state_f); start_g = int(z["g"]) + 1; cur = z["cur"]
        print(f"  resuming {name}_s{seed} at g{start_g}", flush=True)
    if cur is None:
        cur = TRAIN[rng.choice(len(TRAIN), NSAMP, replace=False)].copy()
    for g in range(start_g, G):
        grng = np.random.default_rng(seed * 1000 + g)
        nreal = int(round(LAM * NSAMP))
        real = TRAIN[grng.choice(len(TRAIN), nreal, replace=False)]
        si = grng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        if fixed:
            pool = axmatch_bidir(pool, stale)
        net = train_net(pool, seed * 100 + g)
        cur = std_sample(net, NSAMP, 0.02, seed * 7 + g)
        if fixed:
            cur = axmatch_bidir(cur, stale)
        om = offman(cur, TESTREF)
        save_grid(cur, os.path.join(GRID_DIR, f"{name}_s{seed}_g{g}.png"))
        np.savez(state_f, g=g, cur=cur)
        log(dict(key=f"{name}_s{seed}_g{g}", seed=seed, fixed=fixed, g=g,
                 offman=round(om, 4)))
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
        print(f"  seed{seed} {'FIX' if fixed else 'UNF'} g{g}: offman={om:.4f}  ({el()})",
              flush=True)

for seed in SEEDS:                       # seed-major: seed 0 gives usable result first
    for fixed in [False, True]:
        name = "fixed" if fixed else "unfixed"
        if f"{name}_s{seed}_g{G-1}" in done:
            print(f"{name}_s{seed} complete, skip", flush=True); continue
        recursion(seed, fixed)

print(f"ALL DONE {el()}  (true={TRUE:.4f})", flush=True)
