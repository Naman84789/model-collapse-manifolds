"""CLOSED FORMS for the collapse floor (sigma->0 limit) — permanent verification.

Derived 2026-07-04 from the limit ODE  -dW/du = 1 - (2 m(u)/sqrt(u)) W :

UNCONDITIONAL class (kappa_hat <= kbar only; m = rho/2 everywhere):
    integrating factor e^{-2 rho sqrt(u)};  W(0+) = int_0^inf e^{-2 rho sqrt(u)} du
    = 1/(2 rho^2)   EXACT for every rho.        Floor:  Phi -> sigma^2/(2rho^2) = 1/(8 kbar^2).
    Crossing g=1 at rho_1 = 1/sqrt(2).

NO-OVERSHOOT class (kappa_hat <= min(kappa*, kbar); m = min(sqrt(u)/(1+u), rho/2)):
    band edges are roots of rho x^2 - 2x + rho = 0:  x_-+ = (1 -+ q)/rho, q = sqrt(1-rho^2);
    above band: exact slave W = 1+u;  in band: integrating factor with antiderivative
    F(u) = -(2 rho sqrt(u)+1) e^{-2 rho sqrt(u)} / (2 rho^2);  below band: exact general
    solution W = (1+u) + A(1+u)^2. Assembled:
        W_exit = (1+x_+^2) e^{-4q} + [ (3-2q) - (3+2q) e^{-4q} ] / (2 rho^2)
        g(rho) = 1 + [ W_exit - (1+x_-^2) ] / (1+x_-^2)^2
    Asymptote: g ~ C0/rho^2,  C0 = (1+3e^{-4})/2 = 0.527474;
    Phi -> (1+3e^{-4})/8 / kbar^2 = 0.131868 / kbar^2  (sigma-independent).

This script checks the closed forms against direct numerical integration of the limit ODE.
"""
import numpy as np


def g_closed(rho):
    if rho >= 1.0:
        return 1.0  # band empty: slave all the way, W(0) = 1
    q = np.sqrt(1.0 - rho * rho)
    xlo = (1.0 - q) / rho
    xhi = (1.0 + q) / rho
    ulo = xlo * xlo
    uhi = xhi * xhi
    E = np.exp(-4.0 * q)
    W_exit = (1.0 + uhi) * E + ((3.0 - 2.0 * q) - (3.0 + 2.0 * q) * E) / (2.0 * rho * rho)
    return 1.0 + (W_exit - (1.0 + ulo)) / (1.0 + ulo) ** 2


def g_const_closed(rho):
    return 1.0 / (2.0 * rho * rho)


def limit_g(rho, cap_mode="min", K=400000, u_end=1e-12):
    u_start = max(400.0, 400.0 / rho ** 2)
    us = np.geomspace(u_start, u_end, K + 1)
    W = 1.0 + u_start
    for i in range(K):
        u = us[i]
        du = us[i] - us[i + 1]
        m = rho / 2 if cap_mode == "const" else min(np.sqrt(u) / (1 + u), rho / 2)
        W = W + (1 - (2 * m / np.sqrt(u)) * W) * du
        if W < 0:
            W = 1e-12
    return W


C0 = (1 + 3 * np.exp(-4.0)) / 2
print(f"C0 = (1+3e^-4)/2 = {C0:.6f}    asymptotic  Phi*kbar^2 = C0/4 = {C0/4:.6f}")
print(f"rho_1 (unconditional floor = sigma^2) = 1/sqrt(2) = {1/np.sqrt(2):.6f}")
print()
print(f"{'rho':>6} | {'g_closed':>10} {'g_numeric':>10} {'rel.err':>9} | "
      f"{'const_closed':>12} {'const_num':>10} {'rel.err':>9}")
maxe1 = maxe2 = 0.0
for rho in (0.10, 0.15, 0.20, 0.30, 0.39, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00):
    gc, gn = g_closed(rho), limit_g(rho)
    cc, cn = g_const_closed(rho), limit_g(rho, cap_mode="const")
    e1 = abs(gc - gn) / gn
    e2 = abs(cc - cn) / cn
    maxe1, maxe2 = max(maxe1, e1), max(maxe2, e2)
    print(f"{rho:>6.2f} | {gc:>10.4f} {gn:>10.4f} {e1:>9.2e} | {cc:>12.4f} {cn:>10.4f} {e2:>9.2e}")
print(f"\nmax rel err: no-overshoot {maxe1:.2e}, unconditional {maxe2:.2e}")
print("PASS" if (maxe1 < 5e-3 and maxe2 < 5e-3) else "FAIL")
