"""
Analyze head_to_head.jsonl (Items 2+3) and overlay the Theorem I(a) prediction.

Per arm: mean +/- std trajectory over 5 seeds. Theory line for arm A (Cambridge annealed,
no matcher): the deterministic recursion fixed point v*_det solving
    v = Phi_det( lambda*sigma^2 + (1-lambda) v ),
with Phi_det(w) from the deficit-floor ODE (deterministic, no fit). Arm A should saturate
AT OR ABOVE v*_det (above by the injection ~30%); arms B,C should sit at ~sigma^2.
Prints a summary table + writes figure-ready arrays to head_to_head_figure.json.
"""
import numpy as np, json, os

J = r"C:\Users\naman\Downloads\metric-audit\head_to_head.jsonl"
SIGMA, LAM = 0.05, 0.5
SIG2 = SIGMA ** 2

def Phi_det(w, kbar=3.9, K=200000, t0=1e-6, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

def det_fixed_point():
    v = 0.01
    for _ in range(60):
        w = LAM * SIG2 + (1 - LAM) * v
        v = Phi_det(w)
    return v

rows = {}
if os.path.exists(J):
    for line in open(J):
        d = json.loads(line)
        if "arm" not in d: continue
        rows.setdefault(d["arm"], []).append(d["traj"])

names = {"A": "Cambridge annealed (no matcher)", "B": "std + match (ours)",
         "C": "jump + match (ours)"}
print(f"sigma^2 = {SIG2}")
fig = {"sigma2": SIG2, "arms": {}}
for arm in ["A", "B", "C"]:
    if arm not in rows: continue
    T = np.array(rows[arm])                     # (seeds, G)
    mean = T.mean(0); sd = T.std(0)
    fig["arms"][arm] = dict(name=names[arm], mean=[round(x,6) for x in mean],
                            std=[round(x,6) for x in sd], nseed=len(T))
    print(f"\n{arm}: {names[arm]}  ({len(T)} seeds)")
    print(f"   g0  = {mean[0]:.5f} +/- {sd[0]:.5f}")
    print(f"   gEND= {mean[-1]:.5f} +/- {sd[-1]:.5f}   = {mean[-1]/SIG2:.2f} sigma^2")

if "A" in rows:
    vstar = det_fixed_point()
    fig["theory_Adet_fixedpoint"] = round(vstar, 6)
    print(f"\nTHEORY (deterministic deficit recursion fixed point, no fit): "
          f"v*_det = {vstar:.5f} = {vstar/SIG2:.2f} sigma^2")
    print("Arm A should saturate AT or slightly ABOVE this (injection adds ~30%).")

json.dump(fig, open(r"C:\Users\naman\Downloads\metric-audit\head_to_head_figure.json", "w"), indent=1)
print("\nwrote head_to_head_figure.json")
