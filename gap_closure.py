"""
GAP CLOSURE — connect the measured capacity-degradation law kbar(w) (already sitting
in poolwidth_probe.jsonl, never joined to the recursion analysis) to the two honest
gaps in the paper:

  GAP A (the 2.4x): det_fp_diagnose iterated the deterministic floor with kbar FIXED
    at the clean-tube value 3.9 -> v*_det = 4.5 sigma^2, vs measured recursion floor
    ~9.8-13 sigma^2.  But the probe shows kbar DEGRADES on fatter training pools:
    kbar ~ A - B*sqrt(w), measured across sqrt(w) in [0.03, 0.13] -- and the recursion
    fixed point sits at sqrt(w) ~ 0.116, INSIDE that measured range. Self-consistent
    iteration v <- Phi_det(w(v), kbar(w(v))) is the compounding channel the paper
    hand-waved ("a network trained on fattened data has a worse slope profile").

  GAP B (Assumption A2's tier): the deficit channel of the concave response law
    (measured -0.68) should be COMPUTED by the same ODE with the measured kbar(w):
    c_deficit(w) = dPhi_det(w, kbar(w))/dw - 1. If it lands near -0.68, and its
    saturation (c -> 0 as rho -> 1) explains the concavity, A2's deficit channel is
    derived-given-kbar(w) rather than a free measured law.

Zero new training. Pure numerics on already-logged measurements. Runs in ~1 min.
"""
import os, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.05; SIG2 = SIGMA ** 2; LAM = 0.5

# ---------------- load the measured capacity law ----------------
rows = []
for line in open(os.path.join(BASE, "poolwidth_probe.jsonl")):
    d = json.loads(line)
    if d.get("key", "").startswith("PW_"):
        rows.append(d)
rows.sort(key=lambda r: r["w"])
ws_m = np.array([r["w"] for r in rows])
kb_m = np.array([r["kbar"] for r in rows])
v_m = np.array([r["v_out"] for r in rows])
sig_m = np.sqrt(ws_m)

A_, B_ = np.polyfit(sig_m, kb_m, 1)[::-1][0], -np.polyfit(sig_m, kb_m, 1)[0]
# polyfit returns [slope, intercept]; recompute cleanly:
slope, intercept = np.polyfit(sig_m, kb_m, 1)
def kbar_of_w(w):
    return intercept + slope * np.sqrt(w)
resid = kb_m - kbar_of_w(ws_m)
print("=== measured capacity law (poolwidth_probe.jsonl) ===")
print(f"  kbar(sqrt w) = {intercept:.3f} {slope:+.3f} * sqrt(w)")
print(f"  points: " + "  ".join(f"({s:.2f},{k:.2f})" for s, k in zip(sig_m, kb_m)))
print(f"  fit residuals: {np.round(resid, 3)}  (max |r| = {np.abs(resid).max():.3f})")

# ---------------- the deterministic floor ODE (identical to det_fp_diagnose) ----------
def Phi_det(w, kbar, t0, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

# closed form g(rho), no-overshoot class (paper eq. 8) for cross-check
def g_closed(rho):
    q = np.sqrt(1 - rho * rho)
    xm, xp = (1 - q) / rho, (1 + q) / rho
    um, up = xm * xm, xp * xp
    Wexit = (1 + up) * np.exp(-4 * q) + ((3 - 2 * q) - (3 + 2 * q) * np.exp(-4 * q)) / (2 * rho * rho)
    return 1 + (Wexit - (1 + um)) / (1 + um) ** 2

# ---------------- GAP A: self-consistent recursion fixed point ----------------
print("\n=== GAP A: recursion fixed point, kbar fixed vs self-consistent ===")
print("  targets: Arm A (annealed, g19) = 9.84 sigma^2; fixed-t0 stochastic 10.8-13.2 sigma^2")
print(f"  {'t0':>8} {'mode':>16} {'v*':>9} {'v*/sig2':>8} {'sqrt(w*)':>9} {'kbar(w*)':>9}")

def fixed_point(t0, selfcons, kb_fixed=3.9, damp=0.5):
    v = 0.01
    for _ in range(200):
        w = LAM * SIG2 + (1 - LAM) * v
        kb = kbar_of_w(w) if selfcons else kb_fixed
        vn = Phi_det(w, kb, t0)
        if abs(vn - v) < 1e-8:
            v = vn; break
        v = damp * v + (1 - damp) * vn
    w = LAM * SIG2 + (1 - LAM) * v
    return v, w, (kbar_of_w(w) if selfcons else kb_fixed)

for t0 in [1e-4, 3e-4, 1e-3]:
    for selfcons in [False, True]:
        v, w, kb = fixed_point(t0, selfcons)
        mode = "self-consistent" if selfcons else "kbar=3.9 fixed"
        print(f"  {t0:>8.0e} {mode:>16} {v:>9.5f} {v/SIG2:>7.2f}x {np.sqrt(w):>9.4f} {kb:>9.2f}")

# sensitivity: capacity-law uncertainty (shift intercept by +/- max residual)
print("\n  sensitivity to the capacity-law fit (t0=1e-4, self-consistent):")
base_int = intercept
for d in [-0.1, 0.0, +0.1]:
    intercept = base_int + d
    v, w, kb = fixed_point(1e-4, True)
    print(f"    kbar-law intercept {base_int:.2f}{d:+.2f}: v* = {v/SIG2:.2f} sigma^2")
intercept = base_int

# is sqrt(w*) inside the measured support?
v, w, kb = fixed_point(1e-4, True)
inside = sig_m.min() <= np.sqrt(w) <= sig_m.max()
print(f"\n  sqrt(w*) = {np.sqrt(w):.4f}  measured support [{sig_m.min():.2f}, {sig_m.max():.2f}]"
      f"  -> {'INSIDE (no extrapolation)' if inside else 'OUTSIDE (extrapolating!)'}")

# ---------------- GAP B: deficit channel derived from the same ODE ----------------
print("\n=== GAP B: A2's deficit channel from the ODE + measured kbar(w) ===")
t0p = 0.005   # the probe's sampling truncation
wg = np.linspace(ws_m.min(), ws_m.max(), 41)
Phi_g = np.array([Phi_det(w, kbar_of_w(w), t0p) for w in wg])
dPhi = np.gradient(Phi_g, wg)
print(f"  {'w':>8} {'sqrt(w)':>8} {'rho':>6} {'dPhi/dw':>8} {'c_deficit':>10}")
for wtest in ws_m:
    i = np.argmin(np.abs(wg - wtest))
    rho = 2 * np.sqrt(wtest) * kbar_of_w(wtest)
    print(f"  {wtest:>8.4f} {np.sqrt(wtest):>8.3f} {rho:>6.3f} {dPhi[i]:>8.3f} {dPhi[i]-1:>10.3f}")
print(f"  mean c_deficit over probe range: {np.mean(dPhi)-1:+.3f}   (paper's measured value: -0.68)")

# concavity / saturation: c_deficit(w) should RISE toward 0 as rho -> 1 (g -> 1)
print("\n  saturation check (why A2 is concave): rho(w) and g(rho) across the range")
for wtest in [ws_m.min(), ws_m.mean(), ws_m.max()]:
    rho = 2 * np.sqrt(wtest) * kbar_of_w(wtest)
    print(f"    w={wtest:.4f}: rho={rho:.3f}  g_closed={g_closed(min(rho,0.999)):.3f}"
          f"  Phi_cf=w*g={wtest*g_closed(min(rho,0.999)):.5f}  Phi_ode={Phi_det(wtest, kbar_of_w(wtest), 1e-6):.5f}")

# injection channel = measured total response minus derived deficit
print("\n  attribution: measured total dv/dw (probe) minus derived deficit = injection")
dv_m = np.gradient(v_m, ws_m)
for i, wtest in enumerate(ws_m):
    j = np.argmin(np.abs(wg - wtest))
    print(f"    w={wtest:.4f}: total={dv_m[i]:.3f}  deficit={dPhi[j]-1:+.3f}"
          f"  -> injection={dv_m[i]-1-(dPhi[j]-1):+.3f}")
print("\nDONE")
