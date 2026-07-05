"""Analytic upgrade of Theorem I(a): verify three DERIVED claims before writing proofs.

(1) LIMIT ODE (universality, proved by rescaling t=sigma^2 u, V=sigma^2 W):
        -dW/du = 1 - (2 m(u)/sqrt(u)) W,   m(u) = min( sqrt(u)/(1+u), rho/2 ),
    with exact slave solution W = 1+u above the band (checked). Claim: the full
    variance-ODE floor Phi/sigma^2 -> g(rho) := W(0+) as sigma->0, rho = 2*sigma*kbar
    fixed. Test: limit-ODE g(rho) vs full-ODE floors at sigma in {0.1,0.05,0.02}.

(2) INNER ODE (rho->0 asymptote, proved by y = rho^2 u, Omega = rho^2 W):
        -dOmega/dy = 1 - Omega/sqrt(y),  band exit y_hi = 4, Omega(4)=4 (no-overshoot).
    Claim: g(rho) ~ C0/rho^2 with C0 = Omega(0+). Prediction: C0 = 4*0.141 = 0.564.

(3) UNCONDITIONAL CLASS (kappa_hat <= kbar ONLY, overshoot allowed): the same full ODE
    with cap_mode="const" (kappa_hat == kbar everywhere). Claims: still universal in rho,
    still ~ C0_const/rho^2 (sigma-independent) with C0_const = Omega(0+) from the SAME
    inner ODE started at Omega(4) = 2 (the const-mode arrival: equilibrium W ~ sqrt(u)/rho
    at band entrance u_hi = 4/rho^2 gives Omega = rho^2 W = rho^2*(2/rho^2) = 2).
    Also: where does g_const(rho) cross 1 (floor > sigma^2 unconditionally for rho below that)?

All pure numpy, seconds-to-minutes. Results printed as a table for PROOFS.md.
"""
import numpy as np

SIG2REF = None  # not used; sigma passed explicitly


# ---------------- full variance ODE (same integrator as deficit_floor_law.py) ---------
def full_floor(kbar, sigma, cap_mode="min", K=600000, tstart=8.0, t0=1e-10):
    sig2 = sigma * sigma

    def kstar(t):
        a2 = np.exp(-t)
        s = np.sqrt(1 - a2)
        return s / (a2 * sig2 + s * s)

    ts = np.geomspace(tstart, t0, K + 1)
    V = 1.0
    for i in range(K):
        t = ts[i]
        dt = ts[i] - ts[i + 1]
        s = np.sqrt(1 - np.exp(-t))
        k = kbar if cap_mode == "const" else min(kstar(t), kbar)
        V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0:
            V = 1e-12
    return V / sig2  # in units of sigma^2


# ---------------- (1) limit ODE in u -------------------------------------------------
def limit_g(rho, cap_mode="min", K=600000, u_end=1e-12):
    u_start = max(400.0, 400.0 / rho ** 2)  # well above band exit 4/rho^2
    us = np.geomspace(u_start, u_end, K + 1)
    W = 1.0 + u_start  # exact slave solution of the uncapped limit ODE
    for i in range(K):
        u = us[i]
        du = us[i] - us[i + 1]
        if cap_mode == "const":
            m = rho / 2
        else:
            m = min(np.sqrt(u) / (1 + u), rho / 2)
        W = W + (1 - (2 * m / np.sqrt(u)) * W) * du
        if W < 0:
            W = 1e-12
    return W


# ---------------- (2) inner (band) ODE in y ------------------------------------------
def inner_C0(omega4, K=600000, y_end=1e-14):
    ys = np.geomspace(4.0, y_end, K + 1)
    Om = omega4
    for i in range(K):
        y = ys[i]
        dy = ys[i] - ys[i + 1]
        Om = Om + (1 - Om / np.sqrt(y)) * dy
        if Om < 0:
            Om = 1e-12
    return Om


print("=" * 78)
print("(0) slave-solution check: uncapped limit ODE, W(u)=1+u must be invariant")
u_start = 400.0
us = np.geomspace(u_start, 1e-6, 200000)
W = 1 + u_start
for i in range(len(us) - 1):
    u = us[i]
    du = us[i] - us[i + 1]
    W = W + (1 - (2 / (1 + u)) * W) * du
print(f"    W(1e-6) = {W:.8f}  vs 1+u = {1 + 1e-6:.8f}   (drift = {abs(W - 1 - 1e-6):.2e})")

print("=" * 78)
print("(1) UNIVERSALITY: limit-ODE g(rho) vs full-ODE floors (no-overshoot / min mode)")
print(f"{'rho':>6} {'g_limit':>9} | " + " | ".join(f"full s={s}" for s in (0.10, 0.05, 0.02)))
for rho in (0.20, 0.30, 0.39, 0.50, 0.70, 1.00):
    gl = limit_g(rho)
    fulls = [full_floor(rho / (2 * s), s) for s in (0.10, 0.05, 0.02)]
    print(f"{rho:>6.2f} {gl:>9.3f} | " + " | ".join(f"{f:>9.3f}" for f in fulls))

print("=" * 78)
print("(2) ASYMPTOTE: inner ODE C0 (predict 0.564) and rho^2*g(rho) -> C0")
C0 = inner_C0(4.0)
print(f"    C0 = Omega(0+ | Omega(4)=4) = {C0:.4f}   (prediction 4*0.141 = 0.564)")
for rho in (0.20, 0.10, 0.05, 0.02):
    print(f"    rho={rho:<5} rho^2*g_limit = {rho ** 2 * limit_g(rho):.4f}")

print("=" * 78)
print("(3) UNCONDITIONAL (kappa_hat <= kbar only; overshoot allowed): const mode")
C0c = inner_C0(2.0)
print(f"    C0_const = Omega(0+ | Omega(4)=2) = {C0c:.4f}")
print(f"{'rho':>6} {'g_const_lim':>11} | " + " | ".join(f"full s={s}" for s in (0.10, 0.05, 0.02))
      + "   (full = const-mode ODE)")
for rho in (0.20, 0.30, 0.39, 0.50, 0.70, 1.00, 1.20):
    gl = limit_g(rho, cap_mode="const")
    fulls = [full_floor(rho / (2 * s), s, cap_mode="const") for s in (0.10, 0.05, 0.02)]
    print(f"{rho:>6.2f} {gl:>11.3f} | " + " | ".join(f"{f:>9.3f}" for f in fulls))
for rho in (0.20, 0.10, 0.05, 0.02):
    print(f"    rho={rho:<5} rho^2*g_const = {rho ** 2 * limit_g(rho, cap_mode='const'):.4f}")
print("    crossing g_const(rho)=1 (floor > sigma^2 unconditionally below this rho):")
lo, hi = 0.3, 1.5
for _ in range(40):
    mid = 0.5 * (lo + hi)
    if limit_g(mid, cap_mode="const") > 1:
        lo = mid
    else:
        hi = mid
print(f"    rho_1 = {0.5 * (lo + hi):.3f}   (kbar_1 = rho_1/(2 sigma))")
print("=" * 78)
print("DONE")
