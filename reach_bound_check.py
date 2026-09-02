"""
Is the curvature correction controlled by the REACH rather than by an essential infimum?

Corollary cor:curved bounds Cov(h, Lap h) by L*V with L the Lipschitz constant of Lap h in
h, and states it for a sphere as (D-1)/a^2 with a the smallest radius the ensemble reaches.
That constant is an essential infimum, which is the corollary's stated weakness.

Eliot Beyler (Beyler & Bach, JMLR 2025), asked whether the LINEAR manifold hypothesis in
their denoising comparison is essential, replied that it is not: the results should extend
to manifolds with reach bounded below (the setting of Azangulov, Deligiannidis and Rousseau,
arXiv 2409.18804) provided the noise level stays below that bound. This checks the geometric
statement that suggestion rests on.

CLAIM. For M of dimension d and codimension 1 with reach tau, the level sets of the signed
distance h have shape operator S(I - hS)^{-1}, and |kappa_i| <= 1/tau, so

    |Lap h|  <=  d / (tau - |h|)          on the tube |h| < tau,                     (1)

and hence, as a function of h,   L = sup |d/dh Lap h|  <=  d / (tau - |h|_max)^2.    (2)

(1) is an EQUALITY for a sphere: tau = R, d = D-1, |h| = |r - R|, giving (D-1)/r. The point
of stating it this way is that tau is a property of the manifold, not of the sampler's
ensemble, and it is the quantity the reach literature already assumes.

TEST. An ellipse, where the reach is NOT any radius and the medial axis is a segment rather
than a point. Semi-axes A > B: curvature is maximal at the ends of the major axis, so
tau = B^2/A, and the medial axis runs along the major axis between the two centres of
curvature, at +-(A^2 - B^2)/A.

Two independent computations of Lap h are compared against each other and against (1):
  analytic : Lap h = kappa/(1 + h kappa), kappa the curvature of M at the projection
  numeric  : finite differences of the true distance function on a grid

Run:  python reach_bound_check.py  ->  reach_bound_check.jsonl
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar

OUT = "reach_bound_check.jsonl"


def ellipse(t, A, B):
    return np.stack([A * np.cos(t), B * np.sin(t)], -1)


def ellipse_curvature(t, A, B):
    """kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^{3/2}, positive for a convex curve."""
    xp, yp = -A * np.sin(t), B * np.cos(t)
    xpp, ypp = -A * np.cos(t), -B * np.sin(t)
    return abs(xp * ypp - yp * xpp) / (xp * xp + yp * yp) ** 1.5


def signed_dist(x, A, B, n=20000):
    """Signed distance to the ellipse (negative inside), plus the projection parameter.
    Coarse scan then a local refine, which is robust near the medial axis where the
    nearest point jumps between two branches."""
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = ellipse(ts, A, B)
    d2 = ((P - x) ** 2).sum(-1)
    i = int(d2.argmin())
    lo, hi = ts[i] - 2 * np.pi / n, ts[i] + 2 * np.pi / n
    r = minimize_scalar(lambda t: ((ellipse(np.array(t), A, B) - x) ** 2).sum(),
                        bounds=(lo, hi), method="bounded")
    t_star = float(r.x)
    dist = float(np.sqrt(((ellipse(np.array(t_star), A, B) - x) ** 2).sum()))
    inside = (x[0] / A) ** 2 + (x[1] / B) ** 2 < 1.0
    return (-dist if inside else dist), t_star


def lap_h_numeric(x, A, B, eps=1e-4):
    """Laplacian of the distance function by central differences."""
    tot = 0.0
    h0, _ = signed_dist(x, A, B)
    for k in range(2):
        e = np.zeros(2); e[k] = eps
        hp, _ = signed_dist(x + e, A, B)
        hm, _ = signed_dist(x - e, A, B)
        tot += (hp - 2 * h0 + hm) / eps ** 2
    return tot


if __name__ == "__main__":
    out = []

    # --- control: the sphere, where (1) should be an equality ---
    print("CONTROL: circle R=2.5 in R^2, tau = R = 2.5, d = 1.  bound (1) should be TIGHT")
    R = 2.5
    print(f"  {'r':>6} {'h':>8} {'Lap h exact':>12} {'bound d/(tau-|h|)':>19} {'ratio':>7}")
    for r in (2.4, 2.0, 1.5, 1.0, 0.6):
        h = r - R
        lap = 1.0 / r                      # (D-1)/r with D=2
        bnd = 1.0 / (R - abs(h))
        print(f"  {r:>6.2f} {h:>8.2f} {lap:>12.5f} {bnd:>19.5f} {lap/bnd:>7.4f}")
        out.append(dict(key=f"CIRCLE_r{r}", geom="circle", R=R, r=r, h=round(h, 4),
                        lap=round(lap, 6), bound=round(bnd, 6), ratio=round(lap / bnd, 6)))

    # --- the real test: an ellipse, reach is not a radius ---
    A, B = 3.0, 2.0
    tau = B * B / A
    med = (A * A - B * B) / A
    print(f"\nTEST: ellipse A={A}, B={B}.  reach tau = B^2/A = {tau:.4f}, "
          f"medial axis = the segment |x| <= {med:.4f} on the major axis")
    print(f"  {'point':>18} {'h':>8} {'Lap h analytic':>15} {'Lap h numeric':>14} "
          f"{'bound':>9} {'holds':>6}")
    pts = [np.array(p, float) for p in
           [(2.0, 0.0), (1.5, 0.0), (1.0, 0.0), (0.5, 0.0), (0.0, 0.0),
            (0.0, 1.5), (0.0, 1.0), (0.0, 0.5), (1.0, 1.0), (3.4, 0.0), (0.0, 2.4)]]
    worst = 0.0; nviol = 0
    for x in pts:
        h, t_star = signed_dist(x, A, B)
        kap = float(ellipse_curvature(np.array(t_star), A, B))
        lap_a = kap / (1.0 + h * kap)
        lap_n = lap_h_numeric(x, A, B)
        ok = abs(h) < tau
        bnd = 1.0 / (tau - abs(h)) if ok else float("inf")
        held = ok and abs(lap_a) <= bnd + 1e-9
        if ok and not held: nviol += 1
        if ok: worst = max(worst, abs(lap_a) / bnd)
        verdict = ("yes" if held else "NO") if ok else "past tau"
        print(f"  {str(tuple(x)):>18} {h:>8.4f} {lap_a:>15.5f} {lap_n:>14.5f} "
              f"{bnd if ok else float('nan'):>9.4f} {verdict:>9}")
        out.append(dict(key=f"ELL_{x[0]}_{x[1]}", geom="ellipse", A=A, B=B,
                        tau=round(tau, 6), h=round(h, 5),
                        lap_analytic=round(lap_a, 6), lap_numeric=round(lap_n, 6),
                        bound=(round(bnd, 6) if ok else None),
                        inside_tube=bool(ok), bound_holds=bool(held)))
    print(f"\n  violations inside the tube |h| < tau : {nviol}")
    print(f"  worst |Lap h| / bound inside the tube: {worst:.4f}")
    print("  (points with |h| >= tau are past the reach, where h is not C^2 and the")
    print("   bound is not claimed; for this ellipse that is the medial-axis segment.)")

    with open(OUT, "w") as f:
        for r_ in out:
            f.write(json.dumps(r_) + "\n")
    print(f"\nwrote {OUT}")
