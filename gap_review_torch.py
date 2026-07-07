"""
R4/R5 of the demolition-review checks (torch half; run with the SYSTEM python that has
torch — the venv python runs gap_review_checks.py's R1-R3).

  R4  kbar error bars: poolwidth_probe computed kbar from ONE net per width (seed 501).
      Train the seed-500 nets, recompute, refit the law per seed, propagate to v*.

  R5  MIXTURE test (the sharpest referee attack): the recursion pool at the fixed point
      is half clean tube + half fattened tube, NOT a single Gaussian tube of matched
      width. Train on the actual mixture; compare measured kbar to the single-tube
      law's prediction at the same total width. Tests the width-only (shape-
      independence) assumption the self-consistent closure silently makes.
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(12)
BASE = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.05; SIG2 = SIGMA ** 2; LAM = 0.5
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000

rows = []
for line in open(os.path.join(BASE, "poolwidth_probe.jsonl")):
    d = json.loads(line)
    if d.get("key", "").startswith("PW_"):
        rows.append(d)
rows.sort(key=lambda r: r["w"])
ws_m = np.array([r["w"] for r in rows]); kb_m = np.array([r["kbar"] for r in rows])
sig_m = np.sqrt(ws_m)

def Phi_det(w, kbar, t0=1e-4, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

def fixed_point(kbar_fn, t0=1e-4, damp=0.5):
    v = 0.01
    for _ in range(200):
        w = LAM * SIG2 + (1 - LAM) * v
        vn = Phi_det(w, float(kbar_fn(w)), t0)
        if abs(vn - v) < 1e-8:
            v = vn; break
        v = damp * v + (1 - damp) * vn
    w = LAM * SIG2 + (1 - LAM) * v
    return v, w

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

def ring(n, sig, rng):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")

def ring_mixture(n, sig_a, sig_b, rng):
    return np.concatenate([ring(n // 2, sig_a, rng), ring(n - n // 2, sig_b, rng)])

def train_on(data_np, seed):
    torch.manual_seed(seed)
    data = torch.tensor(data_np)
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, len(data), (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def kbar_fit(net, sig_for_probe):
    ks = []
    for t in [0.1, 0.05, 0.02, 0.01, 0.005]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t)); rng = np.random.default_rng(7)
        th = rng.uniform(0, 2 * np.pi, 4000)
        rr = a * (R + sig_for_probe * rng.normal(size=4000))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(4000, 2))
        rad = np.linalg.norm(x, axis=1)
        uh = x / rad[:, None]; h = rad - a * R
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((4000, 1), float(t))).numpy() * uh).sum(1)
        ks.append(float(np.cov(h, er)[0, 1] / h.var()))
    return max(ks)

t0_ = time.time()
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
print("=== R4: kbar per width, seed-500 nets (probe logged seed-501 only) ===", flush=True)
kb_500 = []
for j, sig in enumerate(SIGS):
    rng = np.random.default_rng(500)
    net = train_on(ring(N, sig, rng), 500)
    kb = kbar_fit(net, sig)
    kb_500.append(kb)
    print(f"  sig={sig:.2f}: kbar(seed500)={kb:.3f}  vs logged(seed501)={kb_m[j]:.3f}"
          f"   [{(time.time()-t0_)/60:.1f} min]", flush=True)
kb_500 = np.array(kb_500)
sl0, ic0 = np.polyfit(sig_m, kb_m, 1)
sl1, ic1 = np.polyfit(sig_m, kb_500, 1)
print(f"  law seed501: kbar = {ic0:.3f} {sl0:+.3f} sqrt(w)")
print(f"  law seed500: kbar = {ic1:.3f} {sl1:+.3f} sqrt(w)")
for tag, (ic, sl) in [("seed501", (ic0, sl0)), ("seed500", (ic1, sl1)),
                      ("mean-law", ((ic0 + ic1) / 2, (sl0 + sl1) / 2))]:
    v, w = fixed_point(lambda w, ic=ic, sl=sl: ic + sl * np.sqrt(w))
    print(f"  v* under {tag}: {v/SIG2:.2f}x  sqrt(w*)={np.sqrt(w):.4f}", flush=True)

print("\n=== R5: MIXTURE pool at the fixed point (the recursion's actual pool shape) ===", flush=True)
v_star, w_star = fixed_point(lambda w: ic0 + sl0 * np.sqrt(w))
sig_fat = float(np.sqrt(v_star))
print(f"  fixed point: v*={v_star:.5f} -> mixture = half sig=0.05 + half sig={sig_fat:.4f}"
      f"  (total w*={w_star:.5f}, sqrt={np.sqrt(w_star):.4f})")
kb_pred = ic0 + sl0 * np.sqrt(w_star)
kbs_mix = []
for seed in [500, 501]:
    rng = np.random.default_rng(seed)
    net = train_on(ring_mixture(N, SIGMA, sig_fat, rng), seed)
    kb_mix = kbar_fit(net, np.sqrt(w_star))
    kbs_mix.append(kb_mix)
    print(f"  seed {seed}: kbar(mixture) = {kb_mix:.3f}   [{(time.time()-t0_)/60:.1f} min]", flush=True)
print(f"  single-tube law prediction at w*: kbar = {kb_pred:.3f}")
print(f"  mixture measured: {np.mean(kbs_mix):.3f} +/- {np.std(kbs_mix):.3f}"
      f"  (|diff| = {abs(np.mean(kbs_mix)-kb_pred):.3f})")
v_mix, w_mix = fixed_point(lambda w: float(np.mean(kbs_mix)))
print(f"  fixed point holding kbar at the mixture-measured value: v* = {v_mix/SIG2:.2f}x sigma^2")
print("\nDONE", flush=True)
