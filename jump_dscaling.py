"""
e_jump(d): how does the ONE-SHOT jump reconstruction floor grow with ambient dimension?
Curved-embedded S^2 (2-manifold) in d in {8,16,32,64,128}, fixed geometric tube
(noise sigma/sqrt(d-2) -> total normal thickness sigma=0.05 for all d).

Per (d, seed in {0,1}): true-data metric baseline; standard-sampler floor (t0=0.005) raw+matched;
jump floor (t_hat=0.02) raw+matched. Question: (jump_matched - true) vs d, and the
jump-vs-standard advantage vs d. Plus recursion endpoints at d in {8,128} (16,64 already in
grand_harness) to confirm NO compounding across the whole d-range.
Resumable via jump_dscaling.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time
torch.set_num_threads(12)

OUT = r"C:\Users\naman\Downloads\metric-audit\jump_dscaling.jsonl"
R_SPH, T_MAX, SIGMA = 2.5, 8.0, 0.05
NSAMP, STEPS, BATCH, THAT = 4000, 2000, 512, 0.02

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
    a = t * f
    return torch.cat([torch.sin(a), torch.cos(a)], 1)

class Score(nn.Module):
    def __init__(self, D, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, D))
    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))

def train(data, seed):
    torch.manual_seed(seed)
    D = data.shape[1]
    net = Score(D); opt = torch.optim.Adam(net.parameters(), 2e-3)
    dt_ = torch.tensor(data); n = len(dt_)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,)); x0 = dt_[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); sig = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, D); xt = a * x0 + sig * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def reverse_seg(net, x, t_from, t_to, k):
    n = len(x)
    ts = np.linspace(t_from, t_to, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        sig = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            eps = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + eps / sig) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x

def std_sample(net, n, D, t0, seed, k=200):
    torch.manual_seed(seed); x = torch.randn(n, D)
    return reverse_seg(net, x, T_MAX, t0, k).numpy().astype("float32")

def jump_sample(net, n, D, t_hat, seed, k=200):
    torch.manual_seed(seed); x = torch.randn(n, D)
    x = reverse_seg(net, x, T_MAX, t_hat, k)
    s = np.sqrt(1 - np.exp(-t_hat)); a = np.exp(-t_hat / 2)
    with torch.no_grad():
        eps = net(x, torch.full((n, 1), float(t_hat)))
    return ((x - s * eps) / a).numpy().astype("float32")

def knn_idx(pts, ref, k, exclude_self=False):
    d2 = ((pts[:, None, :] - ref[None, :, :]) ** 2).sum(2)
    kk = k + 1 if exclude_self else k
    idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    if exclude_self:
        out = np.empty((len(pts), k), dtype=np.int64)
        for i in range(len(pts)):
            row = idx[i][np.argsort(d2[i, idx[i]])]
            out[i] = row[1:]
        idx = out
    return idx

def frames(pts, ref, k, kdim, exclude_self=False):
    idx = knn_idx(pts, ref, k, exclude_self)
    mus = np.empty_like(pts); tangs = np.empty_like(pts); nors = np.empty_like(pts); Ts = []
    for i in range(len(pts)):
        nb = ref[idx[i]]; mu = nb.mean(0)
        C = np.cov((nb - mu).T)
        w, V = np.linalg.eigh(C)
        T = V[:, -kdim:]; rel = pts[i] - mu; tg = T @ (T.T @ rel)
        mus[i] = mu; tangs[i] = tg; nors[i] = rel - tg; Ts.append(T)
    return mus, tangs, nors, Ts

def match_bidir(pts, ref, k, kdim, rng):
    _, _, nr, _ = frames(ref, ref, k, kdim, exclude_self=True)
    v_ref = float(np.mean((nr ** 2).sum(1)))
    mus, tg, no, Ts = frames(pts, ref, k, kdim)
    v_pts = float(np.mean((no ** 2).sum(1)))
    d = pts.shape[1]
    if v_pts > v_ref:
        out = mus + tg + np.sqrt(v_ref / v_pts) * no
    else:
        xi = rng.normal(size=pts.shape).astype(pts.dtype)
        for i in range(len(pts)):
            T = Ts[i]; xi[i] = xi[i] - T @ (T.T @ xi[i])
        out = mus + tg + no + xi * np.sqrt(max(v_ref - v_pts, 0.0) / max(d - kdim, 1))
    return out.astype("float32")

def feats8(u):
    x, y, z = u[:, 0], u[:, 1], u[:, 2]; a = 0.5
    return np.stack([x, y, z, a * (x * x - y * y) / R_SPH, a * x * y / R_SPH,
                     a * y * z / R_SPH, a * z * x / R_SPH,
                     a * (3 * z * z - R_SPH * R_SPH) / (2 * R_SPH)], 1)

def make_embed(d):
    rng = np.random.default_rng(1000)
    Q, _ = np.linalg.qr(rng.normal(size=(d, 8)))
    return Q.astype("float32")

def fib_sphere(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return (R_SPH * np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)).astype("float32")

def sph_u(n, rng):
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return (R_SPH * u).astype("float32")

def offman(X, ref):
    dmin = np.empty(len(X), dtype="float32")
    for i in range(0, len(X), 250):
        c = X[i:i + 250]
        dd = ((c[:, None, :] - ref[None, :, :]) ** 2).sum(2)
        dmin[i:i + 250] = np.sqrt(dd.min(1))
    return float(np.mean(dmin))

DS = [8, 16, 32, 64, 128]
print(f"{'d':>4} {'seed':>4} {'true':>7} {'std_raw':>8} {'std_mat':>8} {'jmp_raw':>8} {'jmp_mat':>8}", flush=True)
for d in DS:
    Q = make_embed(d)
    REF = (feats8(fib_sphere(20000)) @ Q.T).astype("float32")
    def data(n, rng):
        base = feats8(sph_u(n, rng)) @ Q.T
        return (base + rng.normal(0, SIGMA / np.sqrt(d - 2), (n, d))).astype("float32")
    for seed in [0, 1]:
        key = f"D{d}_s{seed}"
        if key in done:
            print(f"{key} done, skip", flush=True); continue
        rng = np.random.default_rng(seed)
        tr = data(NSAMP, rng)
        net = train(tr, 500 + seed)
        ref2k = tr[:2000]
        true = offman(data(NSAMP, np.random.default_rng(77 + seed)), REF)
        sr = std_sample(net, NSAMP, d, 0.005, 900 + seed)
        sm = match_bidir(sr, ref2k, 64, 2, rng)
        jr = jump_sample(net, NSAMP, d, THAT, 900 + seed)
        jm = match_bidir(jr, ref2k, 64, 2, rng)
        rec = dict(key=key, d=d, seed=seed, true=round(true, 4),
                   std_raw=round(offman(sr, REF), 4), std_mat=round(offman(sm, REF), 4),
                   jmp_raw=round(offman(jr, REF), 4), jmp_mat=round(offman(jm, REF), 4))
        log(rec)
        print(f"{d:>4} {seed:>4} {rec['true']:>7} {rec['std_raw']:>8} {rec['std_mat']:>8} "
              f"{rec['jmp_raw']:>8} {rec['jmp_mat']:>8}   ({el()})", flush=True)

# recursion non-compounding across the full d-range (16,64 already in grand_harness)
def recursion(d, seed, G=8, lam=0.5):
    Q = make_embed(d)
    REF = (feats8(fib_sphere(20000)) @ Q.T).astype("float32")
    rng = np.random.default_rng(seed)
    def data(n, r):
        base = feats8(sph_u(n, r)) @ Q.T
        return (base + r.normal(0, SIGMA / np.sqrt(d - 2), (n, d))).astype("float32")
    stale = data(2000, rng)
    cur = data(NSAMP, rng); traj = []
    for g in range(G):
        nreal = int(round(lam * NSAMP))
        real = data(nreal, rng)
        si = rng.choice(len(cur), NSAMP - nreal, replace=True)
        pool = np.concatenate([real, cur[si]], 0).astype("float32")
        pool = match_bidir(pool, stale, 64, 2, rng)
        net = train(pool, seed * 100 + g)
        cur = jump_sample(net, NSAMP, d, THAT, seed * 7 + g)
        cur = match_bidir(cur, stale, 64, 2, rng)
        traj.append(offman(cur, REF))
    return traj

for d in [8, 128]:
    key = f"REC_d{d}_s0"
    if key in done:
        print(f"{key} done, skip", flush=True); continue
    tr = recursion(d, 0)
    log(dict(key=key, d=d, seed=0, traj=[round(x, 4) for x in tr]))
    print(f"{key}: recursion g0={tr[0]:.4f} g7={tr[-1]:.4f}  ({el()})", flush=True)

print(f"ALL DONE {el()}", flush=True)
