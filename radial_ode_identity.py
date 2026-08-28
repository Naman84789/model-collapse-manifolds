"""
PROOF + CHECK: the curved-geometry ODE residual is exactly (D-1) Cov(r, 1/r).

curved_ode_residual.py found a systematic negative residual on the ring with the right
sign for the radial Ito drift but 5-6x the magnitude of the prediction -(D-1) V / R^2.
That prediction was a LINEARISATION of 1/r about the mean radius. The exact term is

    (D-1) Cov(r, 1/r) = (D-1)(1 - E[r] E[1/r]) = -(D-1)/2 E[(r-r')^2/(r r')]   (r,r' iid)

<= 0 always (Cauchy-Schwarz), and unbounded relative to its linearisation as soon as r
carries mass at small radius.

Derivation, in reverse time tau = -t where the sampler is dx = b dtau + dW:

    dr        = <u,b> dtau + (D-1)/(2r) dtau + <u,dW>     (Ito; |u|=1 so d<M> = dtau)
    dVar/dtau = 2 Cov(r, B) + 1,   B = <u,b> + (D-1)/(2r)
    <u,b>     = r/2 - eps_r/s                              (sampler b = x/2 - epshat/s)
    ==> -dV/dt = (1 - 2 khat/s) V + 1 + (D-1) Cov(r, 1/r)

PART A (analytic, exact).  Brownian motion from the origin in D dimensions has
r = sqrt(tau) * chi_D, hence V = tau (D - c_D^2) with c_D = sqrt2 G((D+1)/2)/G(D/2), so
dV/dtau = D - c_D^2 in closed form.  Here b = 0, so the formula predicts
1 + (D-1)(1 - E[r]E[1/r]).  Equality reduces to the Gamma recursion -- shown symbolically
and checked numerically to machine precision for many D, including non-integer D.

PART B (simulation).  2D OU under the SAME Euler-Maruyama scheme the sampler uses, on a
clean shell and on a shell contaminated with 0.2% of mass near the origin.

Run:  python radial_ode_identity.py
"""
import numpy as np
from mpmath import mp, mpf, gamma as G

mp.dps = 40
D_ = 2
LAM = 0.5


def part_a():
    """Exact check. b = 0 (Brownian motion from the origin) in D dimensions, where
    r = sqrt(tau) * chi_D has a closed-form variance, so the formula has no wiggle room."""
    print("=" * 78)
    print("PART A -- analytic: D-dim BM from the origin, r = sqrt(tau) chi_D, b = 0")
    print("=" * 78)
    print("  closed form   dV/dtau = D - c_D^2,            c_D = sqrt2 G((D+1)/2)/G(D/2)")
    print("  our formula   1 + (D-1)(1 - E[r]E[1/r]),      E[r]E[1/r] = G((D+1)/2)G((D-1)/2)/G(D/2)^2")
    print("  difference reduces to  (D-1) G((D-1)/2) - 2 G((D+1)/2),  i.e. z G(z) = G(z+1)")
    print("  at z = (D-1)/2 -- the Gamma recursion, so it is identically zero.")
    print()
    print(f"  {'D':>7} {'closed form':>22} {'our formula':>22} {'|diff|':>10} {'recursion':>10}")
    worst = mpf(0)
    for Dv in (2, 3, 4, 5, 8, 10, 32, 100, 784, 2.5, 3.7, 1.5, 1.01):
        Dm = mpf(Dv)
        tru = Dm - 2 * G((Dm + 1) / 2) ** 2 / G(Dm / 2) ** 2
        our = 1 + (Dm - 1) * (1 - G((Dm + 1) / 2) * G((Dm - 1) / 2) / G(Dm / 2) ** 2)
        rec = abs((Dm - 1) * G((Dm - 1) / 2) - 2 * G((Dm + 1) / 2))
        e = abs(tru - our); worst = max(worst, e)
        print(f"  {Dv:>7} {mp.nstr(tru,16):>22} {mp.nstr(our,16):>22} "
              f"{mp.nstr(e,3):>10} {mp.nstr(rec,3):>10}")
    print()
    print(f"  worst |closed form - our formula| over all D: {mp.nstr(worst, 3)}  (dps={mp.dps})")

    print()
    print("  for contrast, the LINEARISATION -(D-1) V / E[r]^2 on the same object (tau=1):")
    print(f"  {'D':>7} {'exact term':>16} {'linearised':>16} {'ratio':>9}")
    for Dv in (2, 3, 4, 10, 100):
        Dm = mpf(Dv)
        cD = mpf(2) ** mpf("0.5") * G((Dm + 1) / 2) / G(Dm / 2)
        V = Dm - cD ** 2
        ex = (Dm - 1) * (1 - G((Dm + 1) / 2) * G((Dm - 1) / 2) / G(Dm / 2) ** 2)
        lin = -(Dm - 1) * V / cD ** 2
        print(f"  {Dv:>7} {mp.nstr(ex,10):>16} {mp.nstr(lin,10):>16} {mp.nstr(ex/lin,5):>9}")
    print("  the linearisation is already 2.09x wrong for a plain Rayleigh law in D=2.")
    return worst < mpf("1e-30")


def shell(n, rng, m=2.5, sd=0.25, contam=0.0):
    th = rng.uniform(0, 2 * np.pi, n)
    r = m + sd * rng.normal(size=n)
    if contam > 0:
        k = int(round(contam * n))
        r[:k] = np.abs(0.05 * rng.normal(size=k))
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def moments(x):
    r = np.linalg.norm(x, axis=1)
    u = x / np.maximum(r, 1e-30)[:, None]
    ub = (u * (-LAM * x)).sum(1)
    return dict(V=r.var(), m=r.mean(), inv=(1.0 / np.maximum(r, 1e-30)).mean(),
                cov_ub=np.cov(r, ub)[0, 1])


def run(label, contam, dt, total, N, seed=0):
    """Longer window than a single step, so MC error in Delta V is small next to Delta V."""
    rng = np.random.default_rng(seed)
    x = shell(N, rng, contam=contam)
    nstep = int(round(total / dt))
    V0 = moments(x)["V"]
    acc = np.zeros(3)
    for _ in range(nstep):
        M = moments(x)
        flat = 2 * M["cov_ub"] + 1
        acc += (flat,
                flat - (D_ - 1) * M["V"] / M["m"] ** 2,
                flat + (D_ - 1) * (1 - M["m"] * M["inv"]))
        x += (-LAM * x) * dt + np.sqrt(dt) * rng.normal(size=x.shape)
    V1 = moments(x)["V"]
    meas = (V1 - V0) / total
    flat, lin, exact = acc / nstep
    gap = flat - meas
    print(f"  {label:<30} dt={dt:.1e}  measured {meas:+.5f}   flat {flat:+.5f}"
          f"   lin {lin:+.5f}   EXACT {exact:+.5f}")
    print(f"  {'':30} residual vs flat {-gap:+.5f} | explained by lin "
          f"{100*(flat-lin)/gap:6.1f}% | by exact {100*(flat-exact)/gap:6.1f}%")
    return meas, flat, lin, exact


def part_b():
    print("\n" + "=" * 78)
    print("PART B -- simulation: 2D OU, Euler-Maruyama, same scheme as the sampler")
    print("=" * 78)
    for dt in (1e-4, 2.5e-5):
        run("clean shell", 0.000, dt, total=0.05, N=1_500_000)
    print()
    for dt in (1e-4, 2.5e-5):
        run("shell + 0.2% near origin", 0.002, dt, total=0.05, N=1_500_000)


if __name__ == "__main__":
    ok = part_a()
    part_b()
    print("\n" + "=" * 78)
    print("PART A verdict:", "IDENTITY PROVED AND CONFIRMED" if ok else "FAILED")
    print("=" * 78)
