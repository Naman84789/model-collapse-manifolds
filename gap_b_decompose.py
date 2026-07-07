"""
GAP B, second look — decompose the total derivative dPhi_det/dw into channels.

gap_closure.py computed d/dw [ Phi_det(w, kbar(w)) ] = -0.222 mean and called it a
mismatch vs the paper's measured deficit channel -0.68.  But that total derivative
CONTAINS the capacity-degradation chain term:

    d/dw Phi(w, kbar(w)) = dPhi/dw|_kbar  +  (dPhi/dkbar) * kbar'(w)
         (total, -0.22)     (deficit, ?)      (capacity degradation, >0)

The paper's -0.68 was measured at FIXED capacity (kbar held at the clean-tube 3.9).
If the partial term reproduces -0.68, GAP B closes: A2's deficit channel is derived-
given-kbar, and the residual is the SAME compounding channel that closes GAP A.
The old two-channel attribution (deficit -0.68 / injection +0.73) then refines into
three: deficit + capacity-degradation + injection, each with a mechanism.
"""
import os, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
rows = []
for line in open(os.path.join(BASE, "poolwidth_probe.jsonl")):
    d = json.loads(line)
    if d.get("key", "").startswith("PW_"):
        rows.append(d)
rows.sort(key=lambda r: r["w"])
ws_m = np.array([r["w"] for r in rows])
kb_m = np.array([r["kbar"] for r in rows])
v_m = np.array([r["v_out"] for r in rows])
slope, intercept = np.polyfit(np.sqrt(ws_m), kb_m, 1)
def kbar_of_w(w):
    return intercept + slope * np.sqrt(w)

def Phi_det(w, kbar, t0=0.005, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

print("=== decomposition of d Phi_det/dw at each probe w  (t0 = 0.005, probe truncation) ===")
print(f"  kbar law: {intercept:.3f} {slope:+.3f} sqrt(w)")
print(f"  {'w':>8} {'sqrt(w)':>8} | {'part@kb(w)':>10} {'part@3.9':>9} | {'chain':>7} | {'total':>7} | {'c_def':>7} {'c_cap':>7} {'c_tot':>7}")

tot_list, part_list, part39_list, chain_list = [], [], [], []
for w in ws_m:
    kb = kbar_of_w(w)
    hw = 0.05 * w
    part = (Phi_det(w + hw, kb) - Phi_det(w - hw, kb)) / (2 * hw)       # deficit channel, capacity as-degraded
    part39 = (Phi_det(w + hw, 3.9) - Phi_det(w - hw, 3.9)) / (2 * hw)   # deficit channel, paper's fixed-3.9 convention
    hk = 0.05
    dPhi_dk = (Phi_det(w, kb + hk) - Phi_det(w, kb - hk)) / (2 * hk)    # sensitivity to capacity
    chain = dPhi_dk * (slope / (2 * np.sqrt(w)))                        # capacity-degradation channel
    total = part + chain
    tot_list.append(total); part_list.append(part); part39_list.append(part39); chain_list.append(chain)
    print(f"  {w:>8.4f} {np.sqrt(w):>8.3f} | {part:>10.3f} {part39:>9.3f} | {chain:>+7.3f} | {total:>7.3f} | {part-1:>+7.3f} {chain:>+7.3f} {total-1:>+7.3f}")

print(f"\n  means:  deficit c (part@kb(w) - 1) = {np.mean(part_list)-1:+.3f}"
      f"   deficit c (part@3.9 - 1) = {np.mean(part39_list)-1:+.3f}   [paper measured: -0.68]")
print(f"          capacity-degradation channel = {np.mean(chain_list):+.3f}")
print(f"          total = {np.mean(tot_list)-1:+.3f}   [gap_closure total-gradient gave -0.222 -- consistency check]")

# refined three-channel attribution against the MEASURED total dv/dw
print("\n=== refined attribution: measured total = deficit + capacity-degradation + injection ===")
dv_m = np.gradient(v_m, ws_m)
for i, w in enumerate(ws_m):
    inj = dv_m[i] - tot_list[i]
    print(f"  w={w:.4f}: measured dv/dw={dv_m[i]:+.3f} = deficit {part_list[i]-1:+.3f} "
          f"+ cap-deg {chain_list[i]:+.3f} + injection {inj:+.3f}  (+1)")
inj_mean = np.mean(dv_m) - np.mean(tot_list)
print(f"\n  mean channels: deficit {np.mean(part_list)-1:+.3f}, cap-deg {np.mean(chain_list):+.3f}, "
      f"injection {inj_mean:+.3f}  ->  net c = {np.mean(dv_m)-1:+.3f}")
print("\nDONE")
