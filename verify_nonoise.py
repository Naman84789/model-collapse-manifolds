"""Verify the surprising 'nonoise -> 0' collapse is REAL (not NaN/overflow) and
characterize WHAT it collapses to: thin ring (radius R, zero tube thickness, manifold
intact) vs a single point (total collapse). Deterministic reverse drift, 2 gens."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(12)
R, SIGMA, T_MAX = 2.5, 0.05, 8.0
NSAMP, STEPS, BATCH = 4000, 1500, 512

def temb(t, dim=32):
    f = torch.exp(torch.linspace(0, 5, dim // 2)).to(t)
    return torch.cat([torch.sin(t * f), torch.cos(t * f)], 1)

class Score(nn.Module):
    def __init__(self, D=2, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h),
                                 nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))
    def forward(self, x, t): return self.net(torch.cat([x, temb(t, self.te)], 1))

def ring(n, rng, sig=SIGMA):
    th = rng.uniform(0, 2*np.pi, n); rr = R + sig*rng.normal(size=n)
    return np.stack([rr*np.cos(th), rr*np.sin(th)], 1).astype("float32")

def train(data, seed):
    torch.manual_seed(seed); net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    dt_ = torch.tensor(data); n = len(dt_)
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,)); x0 = dt_[idx]
        t = torch.rand(BATCH, 1)*(T_MAX-1e-3)+1e-3
        a = torch.exp(-t/2); s = torch.sqrt(1-torch.exp(-t)); z = torch.randn(BATCH, 2)
        loss = ((net(a*x0+s*z, t)-z)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def det_sample(net, n, seed, t0=1e-4, k=200):
    torch.manual_seed(seed); x = torch.randn(n, 2); ts = np.geomspace(T_MAX, t0, k+1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i+1]-ts[i]); s = np.sqrt(1-np.exp(-t))
        with torch.no_grad(): e = net(x, torch.full((n, 1), t))
        x = x + (-0.5*x + e/s)*dt
    return x.numpy()

def describe(x, tag):
    rr = np.linalg.norm(x, axis=1)
    nan = np.isnan(x).any() or np.isinf(x).any()
    print(f"[{tag}] NaN/Inf={nan}  |x|: mean={rr.mean():.4f} std={rr.std():.5f} "
          f"min={rr.min():.4f} max={rr.max():.4f}")
    print(f"        centroid=({x[:,0].mean():+.3f},{x[:,1].mean():+.3f})  "
          f"2D-spread: std_x={x[:,0].std():.4f} std_y={x[:,1].std():.4f}  "
          f"rvar={rr.var():.6f}")

rng = np.random.default_rng(0)
cur = ring(NSAMP, rng)
for g in range(3):
    real = ring(NSAMP//2, rng); si = rng.choice(len(cur), NSAMP-len(real), replace=True)
    pool = np.concatenate([real, cur[si]], 0).astype("float32")
    net = train(pool, 100+g); cur = det_sample(net, NSAMP, 7+g)
    describe(cur, f"gen{g} deterministic")
print("\nInterpretation: std~0 on |x| but 2D std~R/sqrt(2) => thin RING (radius R, "
      "tube thickness gone, manifold intact). 2D std~0 too => single POINT.")
