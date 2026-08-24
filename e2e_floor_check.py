"""END-TO-END independent check of the capacity floor (Claude, 2026-08-21).

WHY THIS EXISTS. falsify_floor.py attacks the theorem from the variance ODE onward:
independent integrator, limiting cases, 200 random slope profiles, boundary contraction,
symbolic re-derivation. All of that PRESUPPOSES the ODE

        -dV/dt = (1 - 2 khat/s) V + 1                                            (*)

is the correct model of a slope-capped reverse SDE on a sigma-tube. If (*) were itself
mis-derived, every check in that battery would still pass. This script never uses (*).

WHAT IT DOES INSTEAD. It goes back to the probability model and works forward:

  A. PEAK SLOPE FROM THE DENSITY. For a flat sigma-tube the normal marginal under VP-OU
     is exactly N(0, gamma^2), gamma^2 = a^2 sigma^2 + s^2. So the exact eps-slope is
     kappa(t) = s/gamma^2 with NO approximation. Maximising analytically gives
        u* = sigma^2/(1-sigma^2),  kappa_max = 1/(2 sigma sqrt(1-sigma^2)),
     whose sigma->0 limit is the paper's 1/(2 sigma). Checked against a numerical argmax.
     FALSIFIER: a peak that is not 1/(2 sigma) + O(sigma) kills the mechanism story.

  B. MONTE CARLO REVERSE SDE. Simulate the actual reverse-time VP SDE with an explicitly
     slope-capped score, Euler-Maruyama over particles, and MEASURE the output variance.
     No ODE anywhere. Compare against:
       - the unconditional bound   V/sigma^2 >= 1/(2 rho^2),   rho = 2 sigma kbar
       - the no-overshoot law      g(rho)    (what the realistic min() cap should give)
     FALSIFIER: measured variance below the unconditional bound by more than MC error.

  C. SCHEDULE INDEPENDENCE. Re-run B across truncation times t0 spanning three decades.
     The theorem claims the floor does not depend on the schedule.
     FALSIFIER: systematic drift in V/sigma^2 with t0.

  D. ADVERSARIAL CAP PROFILES. The battery drew 200 RANDOM profiles. Random draws rarely
     produce structured worst cases, so this tries hand-picked pathological ones:
     bang-bang, band-only, inverted, spike, and zero-outside-band. All satisfy khat <= kbar
     and so must respect the unconditional bound.
     FALSIFIER: any profile landing below 1/(2 rho^2).

Reported honestly either way. A FAIL here is a real problem with the paper.
"""
import numpy as np

rng_global = np.random.default_rng(20260821)
PASS, FAIL = "PASS", "**FAIL**"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}", flush=True)


# ---------------------------------------------------------------- tube geometry
def a_of(t):
    return np.exp(-t / 2.0)


def s_of(t):
    return np.sqrt(1.0 - np.exp(-t))


def gamma2(t, sig):
    """Exact normal-marginal variance of a flat sigma-tube under VP-OU."""
    return np.exp(-t) * sig ** 2 + (1.0 - np.exp(-t))


def kappa_true(t, sig):
    """Exact eps-slope d eps*/dx = s/gamma^2. No approximation for a flat tube."""
    return s_of(t) / gamma2(t, sig)


# ================================================================== A. peak slope
print("== A: peak eps-slope derived from the tube density ==")
okA = True
for sig in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
    # numerical argmax over a log-spaced grid in u = s^2 = 1 - e^{-t}.
    # Log spacing matters: u* = sigma^2/(1-sigma^2) is tiny for small sigma, so a
    # linear grid would barely resolve the peak at sigma=0.005.
    u = np.geomspace(1e-12, 1.0 - 1e-12, 400_001)
    t = -np.log1p(-u)
    k = kappa_true(t, sig)
    k_num = k.max()
    u_num = u[np.argmax(k)]

    u_ana = sig ** 2 / (1.0 - sig ** 2)                  # analytic argmax
    k_ana = 1.0 / (2.0 * sig * np.sqrt(1.0 - sig ** 2))  # analytic peak
    k_paper = 1.0 / (2.0 * sig)                          # paper's singular-limit claim

    rel_ana = abs(k_num - k_ana) / k_ana
    rel_paper = abs(k_num - k_paper) / k_paper
    ok = rel_ana < 2e-6
    okA &= ok
    print(f"   sigma={sig:<6g} peak_num={k_num:12.4f}  exact={k_ana:12.4f} "
          f"(rel {rel_ana:.2e})   1/(2sigma)={k_paper:12.4f} (rel {rel_paper:.3e})"
          f"   u*_num={u_num:.6g} u*_exact={u_ana:.6g}")
check("A exact peak slope = 1/(2 sigma sqrt(1-sigma^2)) from the density", okA)
# the paper's 1/(2 sigma) must be the sigma->0 limit, error shrinking like sigma^2
sigs_o = [0.1, 0.05, 0.02, 0.01]
errs = []
for sig in sigs_o:
    k_ana = 1.0 / (2.0 * sig * np.sqrt(1.0 - sig ** 2))
    errs.append(abs(k_ana - 1.0 / (2.0 * sig)) / (1.0 / (2.0 * sig)))
# O(sigma^2) means err(s1)/err(s2) = (s1/s2)^2. The sigma list is NOT uniformly halved,
# so the expected ratio must be computed per step rather than assumed to be 4.
obs = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
exp = [(sigs_o[i] / sigs_o[i + 1]) ** 2 for i in range(len(sigs_o) - 1)]
devs = [abs(o - e) / e for o, e in zip(obs, exp)]
check("A paper's 1/(2 sigma) is the singular limit, relative error = O(sigma^2)",
      all(d < 0.02 for d in devs),
      f"observed {[round(o,2) for o in obs]} vs expected {[round(e,2) for e in exp]} "
      f"(max dev {max(devs):.2%})")


# ================================================================== closed forms
def g_uncond(rho):
    return 1.0 / (2.0 * rho ** 2)


def g_noovershoot(rho):
    q = np.sqrt(1.0 - rho ** 2)
    up = ((1.0 + q) / rho) ** 2
    um = ((1.0 - q) / rho) ** 2
    E = np.exp(-4.0 * q)
    W_exit = (1.0 + up) * E + ((3.0 - 2.0 * q) - (3.0 + 2.0 * q) * E) / (2.0 * rho ** 2)
    return 1.0 + (W_exit - (1.0 + um)) / (1.0 + um) ** 2


# ========================================================= B/C/D. reverse SDE MC
def simulate(sig, kbar, t0, T=12.0, n_part=40_000, n_step=3_000, cap="min", seed=0):
    """Euler-Maruyama on the reverse VP SDE with a slope-capped eps model.

    Reverse SDE (tau = T - t, integrating tau upward):
        dx = [x/2 - khat(t) x / s(t)] dtau + dW,    Var(dW) = dtau
    khat is the MODEL slope, capped at kbar. Nothing here uses the variance ODE.
    Steps are geometric in t so the t->0 boundary layer is resolved.
    """
    rng = np.random.default_rng(seed)
    ts = np.geomspace(T, t0, n_step + 1)          # decreasing t
    x = rng.standard_normal(n_part)               # prior at t=T: gamma^2(T) ~ 1

    for i in range(n_step):
        t_hi, t_lo = ts[i], ts[i + 1]
        dtau = t_hi - t_lo                         # positive
        tm = 0.5 * (t_hi + t_lo)                   # midpoint slope
        s = s_of(tm)
        kt = kappa_true(tm, sig)
        if cap == "min":
            kh = min(kt, kbar)
        elif cap == "const":
            kh = kbar                              # always-binding (unconditional class)
        else:
            kh = cap(tm, sig, kbar, kt)            # adversarial profile
        drift = 0.5 * x - kh * x / s
        x = x + drift * dtau + np.sqrt(dtau) * rng.standard_normal(n_part)
    return float(x.var())


print("\n== B: Monte Carlo reverse SDE (no variance ODE used) ==")
SIG, T0 = 0.02, 1e-6
okB = True
for rho in [0.20, 0.39, 0.55]:
    kbar = rho / (2.0 * SIG)
    V = simulate(SIG, kbar, T0, seed=11)
    ratio = V / SIG ** 2
    bound = g_uncond(rho)
    target = g_noovershoot(rho)
    ok = ratio >= bound * 0.97          # 3% MC/discretisation tolerance
    okB &= ok
    print(f"   rho={rho:<5} kbar={kbar:8.2f}  measured V/sigma^2={ratio:9.4f}"
          f"   uncond bound={bound:8.4f}   no-overshoot law={target:8.4f}"
          f"   (meas/law={ratio/target:.3f})")
check("B measured reverse-SDE variance respects the unconditional floor", okB)

# discretisation control: the measured value must be stable in the step count, else
# the agreement above is an artefact of the integrator rather than the dynamics.
kb_conv = 0.39 / (2.0 * SIG)
conv = [(n, simulate(SIG, kb_conv, T0, n_step=n, seed=77) / SIG ** 2)
        for n in [750, 1500, 3000, 6000]]
for n, v in conv:
    print(f"   n_step={n:<6} V/sigma^2={v:9.4f}")
drift = abs(conv[-1][1] - conv[-2][1]) / conv[-1][1]
check("B step-count convergence: result stable as the integrator refines",
      drift < 0.02, f"change from n_step=3000 to 6000 is {drift:.2%}")

print("\n== C: schedule independence (vary truncation time t0) ==")
# The floor is the t0 -> 0 LIMIT, so t0 must be well inside the boundary layer.
# sigma^2 sets that scale: at sigma=0.02, sigma^2 = 4e-4, so t0 = 1e-4 stops the
# sampler before the layer is entered and is NOT a test of the claim. Only t0 << sigma^2
# probes the limit. Reported both ways so the distinction is visible.
kbar = 0.39 / (2.0 * SIG)
vals = []
print(f"   (sigma^2 = {SIG**2:.1e}; the limit needs t0 << sigma^2)")
for t0 in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    V = simulate(SIG, kbar, t0, seed=22)
    vals.append((t0, V / SIG ** 2))
    tag = "outside layer" if t0 > 0.1 * SIG ** 2 else "in limit"
    print(f"   t0={t0:<8g}  V/sigma^2={V/SIG**2:9.4f}   [{tag}]")
deep = [v for t0, v in vals if t0 <= 0.1 * SIG ** 2]
spread = (max(deep) - min(deep)) / np.mean(deep)
check("C floor independent of truncation schedule once t0 << sigma^2",
      spread < 0.05, f"relative spread over t0 in the limit regime: {spread:.2%} "
                     f"(values {[round(v,3) for v in deep]})")

print("\n== D: adversarial cap profiles (structured worst cases, all khat <= kbar) ==")
RHO = 0.39
kbar_d = RHO / (2.0 * SIG)
bound_d = g_uncond(RHO)


def p_bangbang(tm, sig, kb, kt):
    return kb if int(np.log10(max(tm, 1e-18)) * 3) % 2 == 0 else 0.0


def p_bandonly(tm, sig, kb, kt):
    return kb if 0.2 * sig <= tm <= 20.0 * sig else 0.0


def p_inverted(tm, sig, kb, kt):
    return max(0.0, kb - kt)


def p_spike(tm, sig, kb, kt):
    return kb if abs(np.log10(max(tm, 1e-18)) - np.log10(sig ** 2)) < 0.25 else 0.0


def p_zero_outside(tm, sig, kb, kt):
    return min(kt, kb) if tm <= 1.0 else 0.0


okD = True
for name, prof in [("bang-bang", p_bangbang), ("band-only", p_bandonly),
                   ("inverted", p_inverted), ("spike", p_spike),
                   ("zero-outside-band", p_zero_outside)]:
    V = simulate(SIG, kbar_d, T0, cap=prof, seed=33)
    ratio = V / SIG ** 2
    ok = ratio >= bound_d * 0.97
    okD &= ok
    print(f"   {name:<20} V/sigma^2={ratio:10.4f}   bound={bound_d:8.4f}"
          f"   margin={ratio - bound_d:+9.4f}  {'ok' if ok else 'VIOLATION'}")
check("D no structured adversarial profile breaks the unconditional floor", okD)
print("   NOTE: the profiles above clear the bound by blowing up. With khat=0 the drift is")
print("   +x/2, so variance explodes and the bound is met vacuously. The informative test")
print("   is the EQUALITY case below, where the cap binds everywhere and the unconditional")
print("   law 1/(2 rho^2) is supposed to be attained rather than merely respected.")

print("\n== E: tightness at the equality case (cap binding everywhere) ==")
# Euler-Maruyama biases this case HIGH: the drift coefficient (1/2 - kbar/s) stiffens
# without bound as s -> 0, and the bias grows with kbar. Measured at n_step=3000 the
# rho=0.55 error was 6.6%; refining to n_step=150000 brings it to ~1.9% and it then
# plateaus, bouncing within the Monte Carlo noise of the variance estimator
# (sqrt(2/n_part) = 0.58% std at 60k particles, so ~1.2% at 2 sigma). The tolerance
# below is set to that achievable resolution, NOT loosened until the test passes.
# Deviations are one-sided (always ABOVE the law), so the lower bound is never at risk;
# only attainment is being probed here.
TOL_E = 0.04
okE = True
for rho in [0.20, 0.39, 0.55]:
    kb = rho / (2.0 * SIG)
    # t0 = 1e-8 (t0/sigma^2 = 2.5e-5), deep inside the boundary layer. Section C
    # established in advance that the limit requires t0 << sigma^2; at the shallower
    # T0 = 1e-6 the residual was 4-6% and rho-DEPENDENT. At this depth it collapses to a
    # uniform +1.7% across all rho, i.e. a rho-independent one-sided integrator offset.
    V = simulate(SIG, kb, 1e-8, n_step=12_000, n_part=60_000, cap="const", seed=44)
    ratio = V / SIG ** 2
    pred = g_uncond(rho)
    rel = (ratio - pred) / pred
    ok = abs(rel) < TOL_E
    okE &= ok
    print(f"   rho={rho:<5} measured V/sigma^2={ratio:9.4f}   1/(2 rho^2)={pred:9.4f}"
          f"   rel={rel:+.2%}  {'ok' if ok else 'MISMATCH'}")
check(f"E unconditional law is ATTAINED (tight) when the cap binds everywhere "
      f"[tol {TOL_E:.0%}, set by integrator+MC resolution]", okE)

npass = sum(1 for _, ok in results if ok)
print(f"\nSUMMARY: {npass}/{len(results)} checks passed")
for n, ok in results:
    if not ok:
        print(f"  FAILED: {n}")
