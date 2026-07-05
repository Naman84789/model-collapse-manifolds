"""
R'-probe follow-up: resolve the kappa_hat=1.5 anomaly with the SAME cached net.

Tests (all with rprime_probe_net25.pt — no cross-experiment seed differences):
 1. std_sample K=200 log-spaced to t0 in {0.02, 0.005, 0.001}: measured rvar
    vs the ODE prediction integrated with the measured global kappa_hat(t).
 2. jump at t_hat in {0.02, 0.05, 0.1}: raw rvar. Thin (~1e-3) => global affine fit is
    tail-contaminated, local slope near bulk ~ kappa. Fat (~1.4e-2) => genuine deficit.
 3. Local vs global slope: affine fit restricted to |h| < 0.5*gamma and |h| < 0.25*gamma.
 4. Profile eps_rad(h) on a fiber line at t=0.02 (S-shape diagnosis).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"C:\Users\naman\Downloads\metric-audit\rprime_probe2_25.jsonl"
NET = r"C:\Users\naman\Downloads\metric-audit\rprime_probe_net25.pt"
R, SIGMA, T_MAX = 2.5, 0.05, 8.0

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

net = Score(); net.load_state_dict(torch.load(NET)); net.eval()

def rvar(x):
    return float(np.linalg.norm(x, axis=1).var())

def reverse_log(net, n, t0, k=200, seed=900):
    torch.manual_seed(seed)
    x = torch.randn(n, 2)
    ts = np.geomspace(T_MAX, t0, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x.numpy()

def jump(net, n, t_hat, seed=900, k=200):
    torch.manual_seed(seed)
    x = torch.randn(n, 2)
    ts = np.geomspace(T_MAX, t_hat, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    s = np.sqrt(1 - np.exp(-t_hat)); a = np.exp(-t_hat / 2)
    with torch.no_grad():
        e = net(x, torch.full((n, 1), float(t_hat)))
    return ((x - s * e) / a).numpy()

# 1. sampler floors (same net)
for t0 in [0.02, 0.005, 0.001]:
    xs = reverse_log(net, 4000, t0)
    v = rvar(xs)
    a2 = np.exp(-t0); g2 = a2 * SIGMA ** 2 + (1 - a2)
    log(dict(key=f"STD_t0{t0}", rvar=round(v, 5), gam2=round(g2, 5)))
    print(f"std t0={t0}: rvar={v:.5f}  gam2={g2:.5f}  excess={v-g2:.5f}  ({el()})", flush=True)

# 2. jump floors (same net)
for th in [0.02, 0.05, 0.1]:
    xj = jump(net, 4000, th)
    v = rvar(xj)
    log(dict(key=f"JMP_t{th}", rvar=round(v, 5)))
    print(f"jump t_hat={th}: rvar={v:.5f}  (sigma^2={SIGMA**2})  ({el()})", flush=True)

# 3. local vs global affine slope at several t
print(f"{'t':>6} {'k_flat':>7} {'k_glob':>7} {'k_half':>7} {'k_quar':>7}", flush=True)
for t in [0.1, 0.05, 0.02, 0.01, 0.005]:
    a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t)); g2 = a * a * SIGMA ** 2 + s * s
    r2 = np.random.default_rng(11)
    th = r2.uniform(0, 2 * np.pi, 6000)
    rr = a * (R + SIGMA * r2.normal(size=6000))
    x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * r2.normal(size=(6000, 2))
    rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]; h = rad - a * R
    with torch.no_grad():
        er = (net(torch.tensor(x, dtype=torch.float32),
                  torch.full((6000, 1), float(t))).numpy() * uh).sum(1)
    def fit(mask):
        hh, ee = h[mask], er[mask]
        return float(np.cov(hh, ee)[0, 1] / hh.var())
    kg = fit(np.ones(len(h), bool))
    kh = fit(np.abs(h) < 0.5 * np.sqrt(g2))
    kq = fit(np.abs(h) < 0.25 * np.sqrt(g2))
    log(dict(key=f"SLOPE_t{t}", k_flat=round(s / g2, 3), k_glob=round(kg, 3),
             k_half=round(kh, 3), k_quar=round(kq, 3)))
    print(f"{t:>6} {s/g2:>7.2f} {kg:>7.2f} {kh:>7.2f} {kq:>7.2f}", flush=True)

# 4. fiber profile at t=0.02 (radial slice through the scaled ring)
t = 0.02; a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t)); g = np.sqrt(a*a*SIGMA**2 + s*s)
hg = np.linspace(-3 * g, 3 * g, 41)
pts = np.stack([a * R + hg, np.zeros_like(hg)], 1)
with torch.no_grad():
    ep = net(torch.tensor(pts, dtype=torch.float32),
             torch.full((41, 1), float(t))).numpy()[:, 0]
prof = [(round(float(hh), 4), round(float(ee), 4)) for hh, ee in zip(hg, ep)]
log(dict(key="PROFILE_t0.02", profile=prof))
print("profile t=0.02 (h, eps_rad):", flush=True)
for i in range(0, 41, 4):
    print(f"  h={hg[i]:+.4f}  eps={ep[i]:+.4f}", flush=True)
print(f"ALL DONE {el()}", flush=True)
