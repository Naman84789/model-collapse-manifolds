"""
The capacity floor on CURVED data: a rigorous envelope, and what curvature costs it.

radial_ode_identity.py establishes the exact curved variance ODE. In terms of the signed
distance h to the manifold (|grad h| = 1), in reverse time tau = -t:

    -dV/dt = (1 - 2 khat/s) V + 1 + Cov(h, Lap h) + [Cov(h, <n,x>) - V]

The bracket vanishes for a sphere about the origin and for an affine subspace through it.
Cov(h, Lap h) vanishes identically for a flat normal coordinate (Lap h = 0) -- which is why
the flat ODE is exact there -- and is NEGATIVE on a convex manifold, so the FLAT envelope
is not automatically a lower bound on curved data. Rigorous repair, via the two-point
covariance identity Cov(h,g(h)) = 1/2 E[(h-h')(g(h)-g(h'))]:

    Lap h is L-Lipschitz on the support  ==>  |Cov(h, Lap h)| <= L V
    ==>  -dV/dt >= (1 - 2 kbar/s - L) V + 1

i.e. the SAME envelope integrator with the coefficient shifted by -L. For a d-sphere of
radius R in R^D, Lap h = (D-1)/r, so L = (D-1)/a^2 with a = ess inf r over the sampler's
own ensemble.

The question this answers: does the curvature correction move the floor? 2 kbar/s diverges
as s -> 0 while L is a constant, so the correction should be irrelevant exactly where the
floor is set, and matter only at moderate t. That is what reconciles a LARGE measured ODE
residual with a floor bound that still holds on the ring.

Run:  python curved_floor_envelope.py  ->  curved_floor_envelope.jsonl
"""
import json
import numpy as np
from deficit_floor_law import integrate as flat_integrate

OUT = "curved_floor_envelope.jsonl"


def integrate_curved(t0, kbar, sigma, L=0.0, cap_mode="min", K=600000, tstart=8.0):
    """deficit_floor_law.integrate with the coefficient shifted by the curvature constant L."""
    sig2 = sigma * sigma

    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2)
        return s / (a2 * sig2 + s * s)

    ts = np.geomspace(tstart, t0, K + 1)
    V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]
        s = np.sqrt(1 - np.exp(-t))
        k = kbar if cap_mode == "const" else min(kstar(t), kbar)
        V = V + ((1 - 2 * k / s - L) * V + 1) * dt
        if V < 0:
            V = 1e-12
    return V


if __name__ == "__main__":
    out = []
    sigma = 0.05; sig2 = sigma * sigma
    D = 2; R = 2.5

    print("Reproduction check: L=0 must reproduce deficit_floor_law.integrate exactly")
    for kbar in (3.26, 3.9, 5.0):
        a = flat_integrate(1e-6, kbar, sigma, "min")
        b = integrate_curved(1e-6, kbar, sigma, 0.0, "min")
        print(f"  kbar={kbar:<5} flat {a:.8f}   L=0 {b:.8f}   |diff| {abs(a-b):.2e}")
        assert abs(a - b) < 1e-12, "curved integrator does not reduce to the flat one"
    print("  reduces exactly.\n")

    print(f"Curvature cost to the floor, ring R={R} in R^{D}: L = (D-1)/a^2")
    print("  a = the radius the sampler's ensemble stays above (ess inf r)\n")
    print(f"  {'a':>6} {'L':>8} " + "".join(f"{'kbar=%.2f' % k:>12}" for k in (3.26, 3.9, 5.0)))
    base = {k: flat_integrate(1e-6, k, sigma, "min") for k in (3.26, 3.9, 5.0)}
    print(f"  {'flat':>6} {0.0:>8.4f} "
          + "".join(f"{base[k]/sig2:>11.4f} " for k in (3.26, 3.9, 5.0)))
    for a in (2.4, 2.0, 1.5, 1.0, 0.7, 0.5):
        L = (D - 1) / a ** 2
        row = {}
        for k in (3.26, 3.9, 5.0):
            v = integrate_curved(1e-6, k, sigma, L, "min")
            row[k] = v
        print(f"  {a:>6.2f} {L:>8.4f} "
              + "".join(f"{row[k]/sig2:>11.4f} " for k in (3.26, 3.9, 5.0))
              + "  drop " + "/".join(f"{100*(1-row[k]/base[k]):.2f}%" for k in (3.26, 3.9, 5.0)))
        out.append(dict(key=f"CFE_a{a}", a=a, L=round(L, 5),
                        phi_over_sig2={str(k): round(row[k] / sig2, 5) for k in row},
                        drop_pct={str(k): round(100 * (1 - row[k] / base[k]), 4) for k in row}))
    out.append(dict(key="CFE_flat", a=None, L=0.0,
                    phi_over_sig2={str(k): round(base[k] / sig2, 5) for k in base}))

    print("\nWhere does the curvature term actually matter? compare L against 2 kbar / s(t):")
    print(f"  {'t':>9} {'s(t)':>8} {'2*3.9/s':>10} {'L(a=1.5)':>10} {'L / (2kbar/s)':>15}")
    L15 = (D - 1) / 1.5 ** 2
    for t in (1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5):
        s = np.sqrt(1 - np.exp(-t)); c = 2 * 3.9 / s
        print(f"  {t:>9.0e} {s:>8.4f} {c:>10.2f} {L15:>10.4f} {100*L15/c:>14.3f}%")
        out.append(dict(key=f"CFE_share_t{t}", t=t, s=round(float(s), 6),
                        two_kbar_over_s=round(float(c), 4), L=round(L15, 5),
                        share_pct=round(float(100 * L15 / c), 5)))

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}")
