"""
CAPACITY DOMAINS -- attack on open problem (1): kbar(w) is measured on ring tubes only.

Re-measure the capacity-degradation law kbar(w) on three NEW geometries, same network,
same training protocol, same slope-extraction convention as poolwidth_probe.py (affine
fit at t in {0.1, 0.05, 0.02, 0.01, 0.005}, ceiling = max over t):

  SEG    : straight segment tube in 2D, length = ring circumference (15.7), normal =
           y-axis.  This is the paper's flat-tube idealization made EXACT (zero
           curvature; regression restricted to the middle third to dodge endpoints).
           If degradation needs curvature, SEG shows none.
  SPHERE : S^2 of radius 2.5 in R^3, radial normal (intrinsic dim 2, codim 1).
  RING10 : circle of radius 2.5 in the first two coords of R^10, Gaussian tube in all
           9 normal directions; slope measured on the 8 out-of-plane axes (averaged)
           -- high codimension, where the net must produce the slope 9 ways at once.

PREDICTION (if the law is about representing the local normal score, not about rings):
all three show kbar falling with sqrt(w) at a comparable RELATIVE rate (~25-30% over
sqrt(w): 0.03 -> 0.13), with geometry-specific constants.  Falsifier: a flat SEG with
no degradation (curvature-linked) or RING10 flat (codimension-linked).

One net seed (500) per width x geometry; the ring's seed scatter (~0.3 level, slope
consistent) is documented in gap_protocol_robustness.py.  Resumable.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
OUT = r"C:\Users\naman\Downloads\metric-audit\capacity_domains.jsonl"
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
L = 2 * np.pi * R
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
TS_FIT = [0.1, 0.05, 0.02, 0.01, 0.005]

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
    def __init__(self, D, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))
    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))

def data_seg(n, sig, rng):
    x = rng.uniform(-L / 2, L / 2, n); y = sig * rng.normal(size=n)
    return np.stack([x, y], 1).astype("float32")

def data_sphere(n, sig, rng):
    g = rng.normal(size=(n, 3)); u = g / np.linalg.norm(g, axis=1, keepdims=True)
    rr = R + sig * rng.normal(size=n)
    return (u * rr[:, None]).astype("float32")

def data_ring10(n, sig, rng):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    out = np.zeros((n, 10), dtype="float32")
    out[:, 0] = rr * np.cos(th); out[:, 1] = rr * np.sin(th)
    out[:, 2:] = sig * rng.normal(size=(n, 8))
    return out

GEOMS = {"SEG": (data_seg, 2), "SPHERE": (data_sphere, 3), "RING10": (data_ring10, 10)}

def train(geom, sig, seed):
    maker, D = GEOMS[geom]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(maker(N, sig, rng))
    net = Score(D); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, D); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def slope_at(net, geom, sig, t, n=4000, seed=7):
    """affine normal-slope fit under p_t, per geometry's normal coordinate."""
    maker, D = GEOMS[geom]
    rng = np.random.default_rng(seed)
    a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
    x0 = maker(n, sig, rng)
    x = a * x0 + s * rng.normal(size=(n, D)).astype("float32")
    with torch.no_grad():
        e = net(torch.tensor(x), torch.full((n, 1), float(t))).numpy()
    if geom == "SEG":
        m = np.abs(a * x0[:, 0]) < L / 6          # middle third, dodge endpoints
        h = x[m, 1] - x[m, 1].mean()
        return float(np.cov(h, e[m, 1])[0, 1] / max(h.var(), 1e-12))
    if geom == "SPHERE":
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        h = rad - rad.mean(); er = (e * uh).sum(1)
        return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))
    ks = []                                        # RING10: 8 out-of-plane axes
    for j in range(2, 10):
        h = x[:, j] - x[:, j].mean()
        ks.append(np.cov(h, e[:, j])[0, 1] / max(h.var(), 1e-12))
    return float(np.mean(ks))

print(f"{'geom':>7} {'sig':>5} {'kbar':>6}", flush=True)
res = {}
for geom in GEOMS:
    for sig in SIGS:
        key = f"CD_{geom}_sig{sig}"
        if key in done:
            for line in open(OUT):
                d = json.loads(line)
                if d.get("key") == key:
                    res.setdefault(geom, []).append((sig, d["kbar"]))
            print(f"{key} skip", flush=True); continue
        net = train(geom, sig, 500)
        ks = [slope_at(net, geom, sig, t) for t in TS_FIT]
        kb = float(max(ks))
        log(dict(key=key, geom=geom, sig=sig, kbar=round(kb, 3),
                 k_by_t=[round(k, 3) for k in ks]))
        res.setdefault(geom, []).append((sig, kb))
        print(f"{geom:>7} {sig:>5} {kb:>6.2f}   ({el()})", flush=True)

print("\nLAWS kbar(w) = A - B sqrt(w):", flush=True)
for geom, pts in res.items():
    pts = sorted(pts)
    sq = np.array([np.sqrt(s * s) for s, _ in pts]); kb = np.array([k for _, k in pts])
    Bc, Ac = np.polyfit(sq, kb, 1)
    resid = kb - (Ac + Bc * sq)
    rel = (kb[0] - kb[-1]) / kb[0] * 100
    print(f"  {geom:>7}: kbar = {Ac:.2f} {Bc:+.1f} sqrt(w)   max|res|={np.abs(resid).max():.3f}"
          f"   degradation {kb[0]:.2f} -> {kb[-1]:.2f}  ({rel:.0f}% over range)", flush=True)
    log(dict(key=f"LAW_{geom}", A=round(Ac, 3), B=round(Bc, 3),
             maxres=round(float(np.abs(resid).max()), 3), rel_drop_pct=round(rel, 1)))
print(f"ALL DONE {el()}   [ring reference: 4.43 - 11.6 sqrt(w), 24% drop]", flush=True)
