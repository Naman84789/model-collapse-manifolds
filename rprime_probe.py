"""
R'-probe: directly measure the three ingredients of Assumption R' (PROOFS.md, Prop 2.4)
on the ring tube, instead of inferring them from output shapes.

 (i)  band risk: r^2(t) = residual (post-affine-fit) radial risk of the trained net,
      and the true radial risk vs the exact quadrature score — shape over t.
 (ii) block correlation: correlation time of delta along simulated reverse paths,
      as a fraction nu of t.
 (iii) no over-contraction: kappa_hat(t) <= kappa(t) (affine fit slope vs true slope).

Ring: R=1, sigma=0.05 (sigma^2=0.0025), n=4000, standard 2-d Score net, T_MAX=8.
Exact target via 2-d Gaussian-mixture quadrature (1024 theta x 20 GH), radial reduction.
Resumable via rprime_probe25.jsonl.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"rprime_probe25.jsonl"
NET = r"rprime_probe_net25.pt"
R, SIGMA, N, T_MAX, STEPS, BATCH = 2.5, 0.05, 4000, 8.0, 2000, 512

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
    def __init__(self, D=2, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, D))
    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))

def ring_data(n, rng):
    th = rng.uniform(0, 2 * np.pi, n)
    rr = R + SIGMA * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")

rng = np.random.default_rng(0)
net = Score()
if os.path.exists(NET):
    net.load_state_dict(torch.load(NET))
    print(f"net loaded  ({el()})", flush=True)
else:
    torch.manual_seed(500)
    data = torch.tensor(ring_data(N, rng))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); sig = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + sig * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(net.state_dict(), NET)
    print(f"net trained  ({el()})", flush=True)
net.eval()

# ---- exact radial target eps*_rad(r; t) by quadrature -------------------------------
NTH, NGH = 1024, 20
TH = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
gh_x, gh_w = np.polynomial.hermite.hermgauss(NGH)
ZS = np.sqrt(2.0) * gh_x                       # z-nodes ~ N(0,1)
WZ = gh_w / np.sqrt(np.pi)                     # weights sum to 1

def exact_rad(t, rgrid):
    """eps*_rad on rgrid at diffusion time t (x=(r,0) by symmetry)."""
    a = np.exp(-t / 2); s2 = 1 - np.exp(-t)
    mu_r = a * (R + SIGMA * ZS)                # (NGH,) radii of mixture centers
    cx = np.outer(mu_r, np.cos(TH)).ravel()    # (NGH*NTH,)
    cy = np.outer(mu_r, np.sin(TH)).ravel()
    w = np.repeat(WZ, NTH) / NTH
    out = np.empty(len(rgrid))
    for i0 in range(0, len(rgrid), 256):
        r = rgrid[i0:i0 + 256][:, None]
        ll = -((r - cx[None, :]) ** 2 + cy[None, :] ** 2) / (2 * s2) + np.log(w)[None, :]
        m = ll.max(1, keepdims=True)
        p = np.exp(ll - m)
        pw = p / p.sum(1, keepdims=True)
        mux = (pw * cx[None, :]).sum(1)        # posterior mean, x-component
        out[i0:i0 + 256] = (r[:, 0] - mux) / np.sqrt(s2)   # eps*_rad = (x - mu_bar)/s
    return out

T_GRID = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
NS = 4000

print(f"{'t':>7} {'kappa':>8} {'k_hat':>8} {'ratio':>6} {'r2_res':>8} {'r2_true':>8} "
      f"{'r2_tot':>8} {'gam2':>8}", flush=True)
for t in T_GRID:
    key = f"RISK_t{t}"
    if key in done:
        print(f"{key} done, skip", flush=True); continue
    a = np.exp(-t / 2); s2 = 1 - np.exp(-t); s = np.sqrt(s2)
    gam2 = a * a * SIGMA ** 2 + s2
    # sample x ~ p_t
    r2 = np.random.default_rng(int(t * 1e6) + 7)
    th = r2.uniform(0, 2 * np.pi, NS)
    rr = a * (R + SIGMA * r2.normal(size=NS)) + 0.0
    x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    x += s * r2.normal(size=(NS, 2))
    rad = np.linalg.norm(x, axis=1); uhat = x / rad[:, None]
    h = rad - a * R
    with torch.no_grad():
        e = net(torch.tensor(x, dtype=torch.float32),
                torch.full((NS, 1), float(t))).numpy()
    e_rad = (e * uhat).sum(1)
    # exact target on the sampled radii (interp from grid)
    rg = np.linspace(max(rad.min() - 0.01, 1e-4), rad.max() + 0.01, 2000)
    est = exact_rad(t, rg)
    e_star = np.interp(rad, rg, est)
    # affine fits (Lemma 2.2 decomposition), weighted by the sample = p_t
    kap_flat = s / gam2
    vh = h.var()
    k_hat = np.cov(h, e_rad)[0, 1] / vh
    k_star = np.cov(h, e_star)[0, 1] / vh          # curvature-corrected true slope
    res = e_rad - k_hat * h - (e_rad.mean() - k_hat * h.mean())
    r2_res = float((res ** 2).mean())               # residual field risk (Assumption R)
    r2_true = float(((e_rad - e_star) ** 2).mean()) # total radial risk vs exact target
    r2_tot = float(((e - (e_star[:, None] * uhat
                    + (e - (e * uhat).sum(1, keepdims=True) * uhat))) ** 2).sum(1).mean())
    rec = dict(key=key, t=t, kappa_flat=round(float(kap_flat), 4),
               kappa_star=round(float(k_star), 4), kappa_hat=round(float(k_hat), 4),
               ratio=round(float(k_hat / k_star), 4), r2_res=round(r2_res, 5),
               r2_true=round(r2_true, 5), gam2=round(float(gam2), 5))
    log(rec)
    print(f"{t:>7} {k_star:>8.3f} {k_hat:>8.3f} {k_hat/k_star:>6.3f} {r2_res:>8.4f} "
          f"{r2_true:>8.4f} {'':>8} {gam2:>8.5f}   ({el()})", flush=True)

# ---- (ii) correlation fraction nu along reverse paths --------------------------------
def kb_interp():
    """kappa_hat(t), b(t) on a fine grid for delta extraction along paths."""
    ts = np.geomspace(0.001, 0.5, 40)
    ks, bs = [], []
    for t in ts:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        r3 = np.random.default_rng(99)
        th = r3.uniform(0, 2 * np.pi, 1500)
        rr = a * (R + SIGMA * r3.normal(size=1500))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * r3.normal(size=(1500, 2))
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]; hh = rad - a * R
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((1500, 1), float(t))).numpy() * uh).sum(1)
        k = np.cov(hh, er)[0, 1] / hh.var(); b = er.mean() - k * hh.mean()
        ks.append(k); bs.append(b)
    return ts, np.array(ks), np.array(bs)

if "NU_probe" not in done:
    ts_g, ks_g, bs_g = kb_interp()
    M, K = 400, 500
    tpath = np.geomspace(0.2, 0.002, K)
    torch.manual_seed(1234)
    # start paths at t=0.2 from forward samples
    r4 = np.random.default_rng(4)
    a0 = np.exp(-0.1); s0 = np.sqrt(1 - np.exp(-0.2))
    th = r4.uniform(0, 2 * np.pi, M)
    rr = np.exp(-0.1) * (R + SIGMA * r4.normal(size=M))
    x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s0 * r4.normal(size=(M, 2))
    x = torch.tensor(x, dtype=torch.float32)
    dl = np.zeros((M, K), dtype="float32")
    for i in range(K):
        t = float(tpath[i]); s = np.sqrt(1 - np.exp(-t)); a = np.exp(-t / 2)
        with torch.no_grad():
            e = net(x, torch.full((M, 1), t))
        xn = x.numpy(); rad = np.linalg.norm(xn, axis=1); uh = xn / rad[:, None]
        h = rad - a * R
        k = np.interp(t, ts_g, ks_g); b = np.interp(t, ts_g, bs_g)
        dl[:, i] = (e.numpy() * uh).sum(1) - (k * h + b)
        if i < K - 1:
            dt = float(tpath[i + 1] - tpath[i])   # negative
            x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    nus = {}
    for anchor in [0.1, 0.05, 0.02, 0.01, 0.005]:
        i0 = int(np.argmin(np.abs(tpath - anchor)))
        d0 = dl[:, i0] - dl[:, i0].mean()
        v0 = (d0 ** 2).mean()
        nu = None
        for j in range(i0 + 1, K):
            dj = dl[:, j] - dl[:, j].mean()
            c = (d0 * dj).mean() / np.sqrt(v0 * (dj ** 2).mean() + 1e-12)
            if c < 0.5:
                nu = float((tpath[i0] - tpath[j]) / tpath[i0]); break
        nus[str(anchor)] = round(nu, 4) if nu is not None else None
    log(dict(key="NU_probe", nus=nus))
    print(f"NU_probe: {nus}  ({el()})", flush=True)

print(f"ALL DONE {el()}", flush=True)
