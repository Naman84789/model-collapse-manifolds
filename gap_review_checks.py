"""
DEMOLITION-REVIEW CHECKS on the gap-closure claims (commit 256705c), attacking my own
additions the way a hostile referee would:

  R1  "Why sqrt(w)?" -- fit kbar vs w under FIVE functional forms (sqrt, linear, log,
      quadratic-in-sqrt, and a fit-FREE monotone interpolation of the raw points) and
      recompute the self-consistent fixed point under each. If v* is form-insensitive,
      the sqrt is cosmetic (the fixed point only uses the fit as an interpolant, since
      w* is interior). If v* moves a lot, the closure claim is fragile.

  R2  Honest closure arithmetic -- the paper says "roughly half". Compute the actual
      fraction of the gap closed, linear and logarithmic, at the method's real
      t0=1e-4 (and the 1e-3 sensitivity row), against all three measured targets.

  R3  Form-free chain term -- the Gap-B capacity-degradation channel used kbar'(w)
      from the sqrt fit. Recompute it at interior points from centered finite
      differences of the RAW measured (w, kbar) points (no functional form at all).

  R4  kbar error bars -- poolwidth_probe computed kbar from ONE trained net per width
      (the seed-501 net). Train the seed-500 nets and recompute, giving a 2-sample
      scatter per width; refit the law per seed and propagate to v*.

  R5  THE SHARPEST ATTACK: the recursion pool at the fixed point is a MIXTURE
      (half clean sigma=0.05 tube + half fattened tube), not a single Gaussian tube
      of matched total width w*. The kbar(w) law was measured on single tubes only.
      Train nets on the actual fixed-point mixture and measure kbar directly; compare
      to kbar(w*) from the single-tube law. If they disagree, the self-consistent
      closure rests on an unstated shape-independence assumption that FAILS.
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.05; SIG2 = SIGMA ** 2; LAM = 0.5

rows = []
for line in open(os.path.join(BASE, "poolwidth_probe.jsonl")):
    d = json.loads(line)
    if d.get("key", "").startswith("PW_"):
        rows.append(d)
rows.sort(key=lambda r: r["w"])
ws_m = np.array([r["w"] for r in rows])
kb_m = np.array([r["kbar"] for r in rows])
sig_m = np.sqrt(ws_m)

def Phi_det(w, kbar, t0, K=80000, tstart=8.0):
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

# ---------------- R1: functional-form robustness ----------------
print("=== R1: does the sqrt(w) choice matter for the fixed point? ===")
from scipy.interpolate import PchipInterpolator
forms = {}
c = np.polyfit(sig_m, kb_m, 1);  forms["a+b*sqrt(w)  [paper]"] = (lambda w, c=c: np.polyval(c, np.sqrt(w)), np.polyval(c, sig_m))
c2 = np.polyfit(ws_m, kb_m, 1);  forms["a+b*w        [linear]"] = (lambda w, c=c2: np.polyval(c, w), np.polyval(c2, ws_m))
c3 = np.polyfit(np.log(ws_m), kb_m, 1); forms["a+b*log(w)   [log]"] = (lambda w, c=c3: np.polyval(c, np.log(w)), np.polyval(c3, np.log(ws_m)))
c4 = np.polyfit(sig_m, kb_m, 2);  forms["quadratic in sqrt(w)"] = (lambda w, c=c4: np.polyval(c, np.sqrt(w)), np.polyval(c4, sig_m))
pch = PchipInterpolator(ws_m, kb_m)
def pch_clamped(w, lo=ws_m[0], hi=ws_m[-1]):
    return pch(np.clip(w, lo, hi))
forms["PCHIP (fit-free)"] = (pch_clamped, kb_m.copy())

print(f"  {'form':>22} {'max|resid|':>10} {'v* (t0=1e-4)':>13} {'v*/sig2':>8} {'sqrt(w*)':>9}")
vstars = {}
for name, (fn, pred) in forms.items():
    res = np.max(np.abs(kb_m - pred))
    v, w = fixed_point(fn)
    vstars[name] = v
    print(f"  {name:>22} {res:>10.3f} {v:>13.5f} {v/SIG2:>7.2f}x {np.sqrt(w):>9.4f}")
vals = np.array([v / SIG2 for v in vstars.values()])
print(f"  spread across forms: {vals.min():.2f}x .. {vals.max():.2f}x "
      f"(rel spread {(vals.max()-vals.min())/vals.mean()*100:.1f}%)")

# ---------------- R2: honest closure arithmetic ----------------
print("\n=== R2: how much of the gap is actually closed? ===")
v0 = 4.54  # fixed-kbar fixed point at t0=1e-4, in units of sigma^2 (det_fp_diagnose)
for t0, vsc in [(1e-4, None), (1e-3, None)]:
    v, w = fixed_point(forms["a+b*sqrt(w)  [paper]"][0], t0=t0)
    vsc = v / SIG2
    for target, tag in [(9.84, "Arm A g19"), (11.0, "paper's ~11"), (13.2, "fixed-t0 max")]:
        lin = (vsc - v0) / (target - v0) * 100
        logf = np.log((target / v0) / (target / vsc)) / np.log(target / v0) * 100
        print(f"  t0={t0:.0e}  v*={vsc:.2f}x  target {target:>5.2f}x ({tag:>12}): "
              f"residual factor {target/vsc:.2f}  closed {lin:.0f}% linear / {logf:.0f}% log")

# ---------------- R3: form-free chain term ----------------
print("\n=== R3: capacity-degradation channel WITHOUT the sqrt fit ===")
print("  (centered finite differences of the raw (w, kbar) points, interior only)")
t0p = 0.005
for i in [1, 2, 3]:
    w = ws_m[i]
    dk_dw_raw = (kb_m[i + 1] - kb_m[i - 1]) / (ws_m[i + 1] - ws_m[i - 1])
    hk = 0.05
    dPhi_dk = (Phi_det(w, kb_m[i] + hk, t0p) - Phi_det(w, kb_m[i] - hk, t0p)) / (2 * hk)
    slope, intercept = np.polyfit(sig_m, kb_m, 1)
    dk_dw_fit = slope / (2 * np.sqrt(w))
    print(f"  w={w:.4f}: kbar'(w) raw={dk_dw_raw:+.1f} fit={dk_dw_fit:+.1f}  "
          f"chain raw={dPhi_dk*dk_dw_raw:+.3f} fit={dPhi_dk*dk_dw_fit:+.3f}")

# ---------------- R4 + R5: torch experiments ----------------
print("\n=== R4/R5: seed scatter on kbar + the MIXTURE-POOL test ===")
import torch, torch.nn as nn
torch.set_num_threads(12)
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000

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
    """half clean tube sig_a, half fattened tube sig_b -- the recursion pool's shape"""
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
    """identical estimator to poolwidth_probe.py (probe fibers built at width sig)"""
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
print("  R4: kbar per width, seed-500 nets (probe used seed-501 only):")
kb_500 = []
for sig in SIGS:
    rng = np.random.default_rng(500)
    net = train_on(ring(N, sig, rng), 500)
    kb = kbar_fit(net, sig)
    kb_500.append(kb)
    print(f"    sig={sig:.2f}: kbar(seed500)={kb:.3f}  vs logged(seed501)={kb_m[list(SIGS).index(sig)]:.3f}"
          f"   [{(time.time()-t0_)/60:.1f} min]", flush=True)
kb_500 = np.array(kb_500)
sl0, ic0 = np.polyfit(sig_m, kb_m, 1)
sl1, ic1 = np.polyfit(sig_m, kb_500, 1)
print(f"    law seed501: kbar = {ic0:.3f} {sl0:+.3f} sqrt(w)")
print(f"    law seed500: kbar = {ic1:.3f} {sl1:+.3f} sqrt(w)")
for tag, (ic, sl) in [("seed501", (ic0, sl0)), ("seed500", (ic1, sl1)),
                      ("mean-law", ((ic0 + ic1) / 2, (sl0 + sl1) / 2))]:
    v, w = fixed_point(lambda w, ic=ic, sl=sl: ic + sl * np.sqrt(w))
    print(f"    v* under {tag}: {v/SIG2:.2f}x  sqrt(w*)={np.sqrt(w):.4f}")

print("\n  R5: MIXTURE pool at the fixed point (the recursion's actual pool shape):")
# fixed point under the paper law: v* ~ 6.11 sigma^2 -> w* = 0.5*sig2 + 0.5*v*
v_star, w_star = fixed_point(forms["a+b*sqrt(w)  [paper]"][0])
sig_fat = float(np.sqrt(v_star))
print(f"    fixed point: v*={v_star:.5f} -> mixture = half sig=0.05 + half sig={sig_fat:.4f}"
      f"  (total w*={w_star:.5f}, sqrt={np.sqrt(w_star):.4f})")
slope, intercept = np.polyfit(sig_m, kb_m, 1)
kb_pred = intercept + slope * np.sqrt(w_star)
kbs_mix = []
for seed in [500, 501]:
    rng = np.random.default_rng(seed)
    net = train_on(ring_mixture(N, SIGMA, sig_fat, rng), seed)
    kb_mix = kbar_fit(net, np.sqrt(w_star))
    kbs_mix.append(kb_mix)
    print(f"    seed {seed}: kbar(mixture) = {kb_mix:.3f}   [{(time.time()-t0_)/60:.1f} min]", flush=True)
print(f"    single-tube law prediction at w*: kbar = {kb_pred:.3f}")
print(f"    mixture measured: {np.mean(kbs_mix):.3f} +/- {np.std(kbs_mix):.3f}"
      f"  -> shape-independence {'HOLDS' if abs(np.mean(kbs_mix)-kb_pred) < 0.25 else 'FAILS'}"
      f" (|diff|={abs(np.mean(kbs_mix)-kb_pred):.3f})")
# and the fixed point re-solved with the MIXTURE-measured kbar held fixed:
v_mix, w_mix = fixed_point(lambda w: float(np.mean(kbs_mix)))
print(f"    fixed point with mixture-measured kbar: v* = {v_mix/SIG2:.2f}x sigma^2")
print("\nDONE", flush=True)
