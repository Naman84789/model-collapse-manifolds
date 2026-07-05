"""Falsification battery for Theorem 2 (closed-form capacity floor) + Theorem 1 limits.

Attacks the theorem from directions NOT used to derive or previously verify it:
  T1  independent integrator (RK45 via solve_ivp, x=sqrt(u) substitution, no
      hand-rolled Euler) vs both closed forms across rho
  T2  limiting cases: rho->1- (g->1), rho->0 (g*rho^2 -> C0), kbar large (floor->sigma^2)
  T3  comparison principle Monte Carlo: 200 random slope profiles m(x) <= rho/2,
      including profiles that OVERSHOOT the true slope (the '(U) arbitrary overshoot'
      claim) -- every one must satisfy W(0) >= 1/(2 rho^2) - tol
  T4  boundary-condition insensitivity: the claimed quadratic contraction of the
      boundary layer (start with W(U) and 2*W(U): same W(0))
  T5  finite-sigma ODE -> limit ODE convergence from above at three sigmas (the
      universality Lemma), plus kbar >= kappa_max => floor = sigma^2 exactly
  T6  Theorem 1(iii) algebra at lambda=1 (must equal F(sigma^2)) and divergence
      just below lambda* (fixed point must blow up)
  T7  symbolic re-derivation of W_exit via sympy (fully independent path), if sympy
      is installed

Any FAIL printed below is a potential bug in the paper. Report honestly.
"""
import numpy as np
from scipy.integrate import solve_ivp

PASS, FAIL = "PASS", "**FAIL**"
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((name, ok))
    print(f"[{tag}] {name}  {detail}")

# ---------------------------------------------------------------- closed forms
def g_unconditional(rho):
    return 1.0 / (2.0 * rho ** 2)

def g_noovershoot(rho):
    q = np.sqrt(1.0 - rho ** 2)
    up = ((1.0 + q) / rho) ** 2
    um = ((1.0 - q) / rho) ** 2
    E = np.exp(-4.0 * q)
    Wex = (1.0 + up) * E + ((3.0 - 2.0 * q) - (3.0 + 2.0 * q) * E) / (2.0 * rho ** 2)
    return 1.0 + (Wex - (1.0 + um)) / (1.0 + um) ** 2

# ---------------------------------------------------------------- T1: independent integrator
# x = sqrt(u) substitution kills the u^{-1/2} singularity: dW/dx = -2x + 4 m(x^2) W
def limit_floor(rho, mode, X=400.0, rtol=1e-10, atol=1e-12, W0=None):
    def m(u):
        if mode == "const":
            return rho / 2.0
        # no-overshoot envelope: min(true rescaled slope, cap)
        return np.minimum(np.sqrt(u) / (1.0 + u), rho / 2.0)
    def rhs(x, W):
        return -2.0 * x + 4.0 * m(x * x) * W
    Winit = (1.0 + X * X) if W0 is None else W0
    sol = solve_ivp(rhs, [X, 0.0], [Winit], rtol=rtol, atol=atol, dense_output=False)
    return sol.y[0, -1]

print("== T1: independent RK45 vs closed forms ==")
worst_u = worst_n = 0.0
for rho in [0.1, 0.2, 0.39, 0.5, 0.707, 0.9, 0.99]:
    nu = limit_floor(rho, "const")
    nn = limit_floor(rho, "noov")
    eu = abs(nu - g_unconditional(rho)) / g_unconditional(rho)
    en = abs(nn - g_noovershoot(rho)) / g_noovershoot(rho)
    worst_u, worst_n = max(worst_u, eu), max(worst_n, en)
    print(f"  rho={rho:5.3f}  uncond num={nu:10.5f} cf={g_unconditional(rho):10.5f} "
          f"rel={eu:.2e} | noov num={nn:9.5f} cf={g_noovershoot(rho):9.5f} rel={en:.2e}")
check("T1 unconditional closed form matches independent RK45", worst_u < 1e-6,
      f"worst rel err {worst_u:.2e}")
check("T1 no-overshoot closed form matches independent RK45", worst_n < 1e-6,
      f"worst rel err {worst_n:.2e}")

# ---------------------------------------------------------------- T2: limits
print("== T2: limiting cases ==")
g999 = g_noovershoot(0.999999)
check("T2 rho->1-: g -> 1", abs(g999 - 1.0) < 1e-3, f"g(0.999999)={g999:.6f}")
C0 = (1.0 + 3.0 * np.exp(-4.0)) / 2.0
r = 1e-3
check("T2 rho->0: g*rho^2 -> C0=(1+3e^-4)/2", abs(g_noovershoot(r) * r * r - C0) < 1e-2,
      f"g*rho^2={g_noovershoot(r)*r*r:.6f} vs C0={C0:.6f}")
check("T2 crossing: 1/(2rho^2)=1 exactly at rho=1/sqrt(2)",
      abs(g_unconditional(1/np.sqrt(2)) - 1.0) < 1e-12)
mono = all(g_noovershoot(a) > g_noovershoot(b)
           for a, b in zip(np.linspace(.05,.95,40)[:-1], np.linspace(.05,.95,40)[1:]))
check("T2 g(rho) strictly decreasing on (0,1)", mono)
gt1 = all(g_noovershoot(x) > 1.0 for x in np.linspace(0.05, 0.999, 200))
check("T2 g(rho) > 1 for all rho in (0,1)  [floor exceeds true width]", gt1)

# ---------------------------------------------------------------- T3: comparison principle MC
print("== T3: 200 random profiles m<=rho/2 (incl. overshoot of true slope) ==")
rng = np.random.default_rng(0)
viol = 0; margin = np.inf
rho = 0.39
bound = g_unconditional(rho)
for trial in range(200):
    # random smooth profile in [0, rho/2]; half the trials force overshoot of the
    # true slope at small u (m > sqrt(u)/(1+u) there) to attack claim (U)
    knots = rng.uniform(0, rho / 2, 8)
    if trial % 2:
        knots[:3] = rho / 2  # max contraction at small u = certain overshoot
    xs = np.linspace(0, 20, 8)
    def m(u, xs=xs, ks=knots):
        return np.interp(np.sqrt(u), xs, ks)
    def rhs(x, W):
        return -2.0 * x + 4.0 * m(x * x) * W
    W0 = solve_ivp(rhs, [400.0, 0.0], [1.0 + 400.0 ** 2], rtol=1e-9, atol=1e-11).y[0, -1]
    margin = min(margin, W0 - bound)
    if W0 < bound - 1e-6 * bound:
        viol += 1
check("T3 no violation of W(0) >= 1/(2rho^2) in 200 random/overshoot profiles",
      viol == 0, f"min margin above bound: {margin:.4f} (bound {bound:.4f})")

# ---------------------------------------------------------------- T4: boundary contraction
# The Lemma does NOT claim zero boundary influence; it claims deviations contract as
# ((1+u)/(1+u'))^2 in the uncapped region and e^{-4q} across the band. Test the
# MECHANISM: perturb the boundary by delta=(1+X^2) and compare the residual at 0
# against the theorem's own predicted contraction factor.
print("== T4: boundary perturbation contracts at the theorem's predicted rate ==")
rho_t4, X_t4 = 0.39, 400.0
a = limit_floor(rho_t4, "noov")
b = limit_floor(rho_t4, "noov", W0=2.0 * (1.0 + X_t4 ** 2))
q4 = np.sqrt(1 - rho_t4 ** 2)
up4 = ((1 + q4) / rho_t4) ** 2
um4 = ((1 - q4) / rho_t4) ** 2
delta = 1.0 + X_t4 ** 2
pred = delta * ((1 + up4) / (1 + X_t4 ** 2)) ** 2 * np.exp(-4 * q4) / (1 + um4) ** 2
obs = abs(b - a)
check("T4 boundary residual matches predicted quadratic contraction (within 5%)",
      abs(obs / pred - 1.0) < 0.05,
      f"observed {obs:.3e} vs predicted {pred:.3e} (ratio {obs/pred:.3f})")

# ---------------------------------------------------------------- T5: finite-sigma ODE
print("== T5: finite-sigma variance ODE ==")
def finite_sigma_floor(sigma, kbar, t0=1e-10, T=8.0):
    def rhs(t, V):
        s2 = 1.0 - np.exp(-t)
        s = np.sqrt(s2)
        g2 = np.exp(-t) * sigma ** 2 + s2
        khat = min(s / g2, kbar)
        return -( (1.0 - 2.0 * khat / s) * V + 1.0 )  # dV/dt going down = -RHS... solve_ivp t: T->t0
    sol = solve_ivp(rhs, [T, t0], [np.exp(-T) * sigma ** 2 + (1 - np.exp(-T))],
                    rtol=1e-10, atol=1e-14)
    return sol.y[0, -1]

vals = []
for sig in [0.10, 0.05, 0.02]:
    kbar = 0.39 / (2 * sig)          # fixed rho = 0.39
    v = finite_sigma_floor(sig, kbar) / sig ** 2
    vals.append(v)
    print(f"  sigma={sig:.2f}  V(t0)/sigma^2 = {v:.4f}")
lim = limit_floor(0.39, "noov")
conv = vals[0] > vals[1] > vals[2] > lim and abs(vals[2] - lim) / lim < 0.02
check("T5 finite-sigma floors decrease toward the limit value from above", conv,
      f"{vals[0]:.3f} > {vals[1]:.3f} > {vals[2]:.3f} -> limit {lim:.3f}")
sig = 0.05
v_uncapped = finite_sigma_floor(sig, kbar=1e9) / sig ** 2
check("T5 kbar -> infinity: floor -> sigma^2 (cap never binds, V=gamma^2)",
      abs(v_uncapped - 1.0) < 5e-3, f"V/sigma^2 = {v_uncapped:.5f}")

# ---------------------------------------------------------------- T6: Theorem 1(iii) algebra
print("== T6: law-of-collapse limits ==")
sig2, c, e0sq = 0.0025, 0.05, 0.0001
v_inf = lambda lam: ((1 + c) * lam * sig2 + e0sq) / (lam * (1 + c) - c)
F = lambda w: w + e0sq + c * w           # F(w) = w + e^2(w), e^2 = e0^2 + c w
check("T6 lambda=1: v_inf equals F(sigma^2) (all-fresh-data self-consistency)",
      abs(v_inf(1.0) - F(sig2)) < 1e-15, f"{v_inf(1.0):.6f} vs {F(sig2):.6f}")
lam_star = c / (1 + c)
check("T6 lambda -> lambda*+: fixed point diverges",
      v_inf(lam_star * 1.001) > 100 * v_inf(0.5),
      f"v_inf(1.001*lambda*)={v_inf(lam_star*1.001):.3f} vs v_inf(0.5)={v_inf(0.5):.5f}")

# ---------------------------------------------------------------- T7: symbolic re-derivation
print("== T7: symbolic band crossing (sympy) ==")
try:
    import sympy as sp
    x, rho_s, q_s = sp.symbols("x rho q", positive=True)
    # ODE across the band in x=sqrt(u): dW/dx = -2x + 2 rho W  (m = rho/2)
    # integrating factor e^{-2 rho x}; particular from F(x) = -(2 rho x + 1) e^{-2 rho x}/(2 rho^2)
    W = sp.Function("W")
    xm = (1 - q_s) / rho_s
    xp = (1 + q_s) / rho_s
    sol = sp.dsolve(sp.Eq(W(x).diff(x), -2 * x + 2 * rho_s * W(x)), W(x),
                    ics={W(xp): 1 + xp ** 2})
    Wexit_sym = sp.simplify(sol.rhs.subs(x, xm))
    # paper's closed form
    up = xp ** 2
    Wexit_paper = (1 + up) * sp.exp(-4 * q_s) \
        + ((3 - 2 * q_s) - (3 + 2 * q_s) * sp.exp(-4 * q_s)) / (2 * rho_s ** 2)
    diff = sp.simplify((Wexit_sym - Wexit_paper).subs(q_s, sp.sqrt(1 - rho_s ** 2)))
    num = [complex(diff.subs(rho_s, rv)) for rv in (0.2, 0.5, 0.8)]
    ok = all(abs(z) < 1e-9 for z in num)
    check("T7 sympy dsolve band crossing == paper W_exit (symbolic, independent)",
          ok, f"residuals at rho=0.2/0.5/0.8: {[f'{abs(z):.1e}' for z in num]}")
except ImportError:
    print("  sympy not installed; skipped (T1 already provides an independent path)")

# ---------------------------------------------------------------- summary
print()
nfail = sum(1 for _, ok in results if not ok)
print(f"SUMMARY: {len(results) - nfail}/{len(results)} checks passed"
      + ("" if nfail == 0 else f"  <<< {nfail} FAILURES - investigate before submission"))
