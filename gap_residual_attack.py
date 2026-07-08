"""
Attack the REMAINING 1.5-1.8x residual of the recursion gap with measured inputs only.

  N1  Jensen/fluctuation correction: the recursion iterates a NOISY map, and the
      deterministic fixed point of the mean map is biased low if Phi is convex in w.
      Shift ~ 0.5 * Phi''(w*) * (1-lam)^2 * Var(v_g), with Var(v_g) measured from the
      fixed-t0 stochastic arms' plateau (inject_ablation.jsonl, gens 5-14, 3 seeds).
      Either this closes part of the residual or it rules fluctuations OUT.

  N2  Measured-tracking profile: Phi_det idealizes the network as PERFECT tracking
      below its ceiling, khat = min(kstar, kbar). The rprime probe MEASURED tracking
      at 0.95-0.97 of the true slope below the ceiling (paper, "measured network
      profile"). Model khat(t) = min(tau*kstar(t), kbar(w)) with tau in {1.0, 0.97,
      0.95} and recompute the self-consistent fixed point under both probe seeds'
      capacity fits. Zero new parameters: tau and kbar(w) are both already-published
      measurements.
"""
import os, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.05; SIG2 = SIGMA ** 2; LAM = 0.5

rows = []
for line in open(os.path.join(BASE, "poolwidth_probe.jsonl")):
    d = json.loads(line)
    if d.get("key", "").startswith("PW_"):
        rows.append(d)
rows.sort(key=lambda r: r["w"])
ws_m = np.array([r["w"] for r in rows]); kb_m = np.array([r["kbar"] for r in rows])
sig_m = np.sqrt(ws_m)
sl501, ic501 = np.polyfit(sig_m, kb_m, 1)
# seed-500 law measured in gap_review_torch.py:
ic500, sl500 = 4.056, -9.688

def Phi_det(w, kbar, t0=1e-4, tau=1.0, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(tau * kstar(t), kbar)
        V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

def fixed_point(ic, sl, tau=1.0, t0=1e-4, damp=0.5):
    v = 0.01
    for _ in range(200):
        w = LAM * SIG2 + (1 - LAM) * v
        vn = Phi_det(w, ic + sl * np.sqrt(w), t0, tau)
        if abs(vn - v) < 1e-8:
            v = vn; break
        v = damp * v + (1 - damp) * vn
    w = LAM * SIG2 + (1 - LAM) * v
    return v, w

# ---------------- N1: fluctuation (Jensen) correction ----------------
print("=== N1: is the residual just fluctuation bias? ===")
trajs = []
for line in open(os.path.join(BASE, "inject_ablation.jsonl")):
    d = json.loads(line)
    if d.get("noise") is True:
        trajs.append(d["traj"])
plateau = np.array([t[5:] for t in trajs])        # gens 5-14, 3 seeds
var_v = np.mean([np.var(p) for p in plateau])     # within-seed gen-to-gen variance
print(f"  fixed-t0 stochastic plateau (g5-14): mean v = {plateau.mean():.4f} "
      f"({plateau.mean()/SIG2:.1f} sigma^2), within-seed Var(v_g) = {var_v:.2e}")
v_star, w_star = fixed_point(ic501, sl501)
h = 0.10 * w_star
Phi_m = Phi_det(w_star - h, ic501 + sl501 * np.sqrt(w_star - h))
Phi_0 = Phi_det(w_star,     ic501 + sl501 * np.sqrt(w_star))
Phi_p = Phi_det(w_star + h, ic501 + sl501 * np.sqrt(w_star + h))
Phi_pp = (Phi_p - 2 * Phi_0 + Phi_m) / h**2
shift = 0.5 * Phi_pp * (1 - LAM) ** 2 * var_v
print(f"  Phi''(w*) = {Phi_pp:.1f}   Jensen shift = 0.5*Phi''*(1-lam)^2*Var(v) "
      f"= {shift:.2e} = {shift/SIG2:.3f} sigma^2")
print(f"  -> fluctuation bias {'CANNOT explain the residual' if abs(shift)/SIG2 < 0.5 else 'is significant'}"
      f" (residual is 3-6 sigma^2)")

# ---------------- N2: measured tracking factor ----------------
print("\n=== N2: fixed point with the MEASURED tracking lag (tau = 0.95-0.97) ===")
print("  khat(t) = min(tau*kstar(t), kbar(w));  tau and kbar(w) are published measurements")
print(f"  {'law':>8} {'tau':>5} {'v*':>9} {'v*/sig2':>8} {'sqrt(w*)':>9} | residual vs 9.84 / 11.0 / 13.2")
for tag, ic, sl in [("seed501", ic501, sl501), ("seed500", ic500, sl500)]:
    for tau in [1.00, 0.97, 0.95]:
        v, w = fixed_point(ic, sl, tau=tau)
        r = [t / (v / SIG2) for t in (9.84, 11.0, 13.2)]
        inside = "in" if np.sqrt(w) <= sig_m.max() else "OUT"
        print(f"  {tag:>8} {tau:>5.2f} {v:>9.5f} {v/SIG2:>7.2f}x {np.sqrt(w):>9.4f} ({inside})"
              f" | {r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f}")
print("\nDONE")
