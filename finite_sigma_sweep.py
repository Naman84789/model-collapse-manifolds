"""Finite-sigma no-overshoot variance-ODE floor as a function of rho, for fig0(b).

Integrates the exact variance ODE (Lemma 'variance ODE' in the paper)
    -dV/dt = (1 - 2*khat(t)/s(t)) V + 1,   khat = min(kappa(t), kbar)   [no-overshoot]
from t=T down to t0->0, V(T)=gamma^2(T), and reports V(t0)/sigma^2.

Self-check: must reproduce the appendix values at sigma=0.05
    rho=0.20 (kbar=2.0)  -> 14.12
    rho=0.39 (kbar=3.9)  -> 3.81
If those match, the rho-sweep is trustworthy.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp

SIG = 0.05
T, T0 = 12.0, 1e-7


def floor_at(rho, sig=SIG):
    kbar = rho / (2 * sig)

    def kappa(t):
        s2 = 1 - np.exp(-t)
        g2 = np.exp(-t) * sig ** 2 + s2
        return np.sqrt(s2) / g2

    def rhs(t, V):
        s = np.sqrt(1 - np.exp(-t))
        khat = min(kappa(t), kbar)
        return -((1 - 2 * khat / s) * V + 1.0)

    g2T = np.exp(-T) * sig ** 2 + (1 - np.exp(-T))
    sol = solve_ivp(rhs, (T, T0), [g2T], method="Radau",
                    rtol=1e-9, atol=1e-12, dense_output=False)
    return sol.y[0, -1] / sig ** 2


# ---- self-check against the appendix ----
chk = {0.20: 14.12, 0.39: 3.81}
ok = True
for rho, expect in chk.items():
    got = floor_at(rho)
    rel = abs(got - expect) / expect
    flag = "OK" if rel < 0.03 else "MISMATCH"
    if rel >= 0.03:
        ok = False
    print(f"  rho={rho}: ODE={got:.3f}  appendix={expect}  rel={rel:.3%}  {flag}")

if ok:
    rhos = np.array([0.10, 0.15, 0.20, 0.28, 0.39, 0.55, 0.75, 0.95])
    vals = [float(floor_at(r)) for r in rhos]
    out = {"sigma": SIG, "rho": rhos.tolist(), "floor_over_sig2": vals}
    with open("finite_sigma_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("SWEEP:", [f"{r:.2f}->{v:.2f}" for r, v in zip(rhos, vals)])
    print("wrote finite_sigma_sweep.json")
else:
    print("SELF-CHECK FAILED -- do not use the sweep")
