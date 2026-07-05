# PROOFS — Manifold model collapse: the law, the floor, the fix

Working manuscript-grade proofs, 2026-07-03/04. Companion to THEORY_NOTE.md (statements,
strategies, challenge log) and SOLUTION_NOTE.md (experiments). Everything here is stated
in the σ-tube model; each assumption is tagged with the harness arm that validates it.

Status legend: [PROVED] complete proof below · [STATED] statement fixed, proof in progress
· [PLANNED] section stub.

**Claim tiers (use these verbs in the manuscript, everywhere).** Every load-bearing
sentence belongs to exactly one tier:
- **"We prove"** — unconditional mathematics given the stated assumptions (Lemmas 0.1,
  2.1, 2.2, 3.0–3.3; Theorems I(a), II′, I(c′); Prop 2.3). The floor law is CLOSED FORM
  in the σ→0 limit (Lemma 3.3): g(ρ) explicit, asymptotic constants (1+3e^{−4})/8
  (no-overshoot) and exactly 1/8 (unconditional); finite-σ deviations are verified
  numerically (1–8% at σ = 0.05) — no data fit anywhere.
- **"We observe"** — experimentally validated, multi-seed, primary-source-audited
  (head-to-head 9.84σ² vs 1.01σ²; two-horned ablation; pixel-MNIST 3.06 vs 1.66; the
  zero-fit single-pass floor match; d-scaling saturation; n-flatness).
- **"We model"** — mechanism captured by an assumed structure that is measured but not
  derived ((A0) scalar reduction; (A2′) concave response; (R′) block model for the
  residual channel; the 2.4× recursion compounding gap, bounded but not computed).
The ledger in §5 assigns every ingredient to a tier; do not promote across tiers in prose.

---

## §0. Framework [PROVED — definitions and exact elementary facts]

### 0.1 Data model (σ-tube)

M ⊂ R^d a compact smooth k-dimensional manifold with reach τ > 0. Data
X₀ = φ(U) + σ N_⊥, with U uniform on M, N_⊥ standard normal in the (d−k)-dimensional
normal space at φ(U), σ > 0 the tube thickness. The singular regime is σ → 0; Fisher
information of the marginal scales as J ≍ (d−k)/σ² → ∞ (this is Case 2 of 2606.13796).

### 0.2 Forward process and normal coordinate

VP-OU forward: X_t = a(t) X₀ + s(t) Z, a(t) = e^{−t/2}, s²(t) = 1 − e^{−t}, Z ~ N(0, I_d).

**Flat-tube (normal-coordinate) idealization.** Fix p ∈ M and work in a normal fiber,
treating the fiber marginal as exactly Gaussian and decoupled from tangential position.
All displayed formulas are exact in this idealization; curvature enters as O(s²/τ)
corrections (tracked explicitly where they matter, §4). In a normal coordinate x ∈ R
(one normal direction; the (d−k) directions decouple):

    X₀ ~ N(0, σ²),   X_t ~ N(0, γ²(t)),   γ²(t) := a²(t) σ² + s²(t).

**True ε-predictor and score.** With ε*(x,t) := E[Z | X_t = x] (the DSM target):

    ε*(x,t) = κ(t) · x,    κ(t) = s(t) / γ²(t),    ∇ log p_t(x) = −x/γ²(t) = −ε*/s.

κ(t) is the slope the estimator must represent. κ(t) ≈ 1/s(t) for s² ≫ σ², peaks at
κ_max ≈ 1/(2σ) near s ≈ σ, and κ(0⁺) = 0⁺/σ² → κ(t)→ t^{1/2}/σ² for small t. In the
singular limit σ → 0 the required slope diverges — the precise sense in which "infinite
Fisher information" bites the estimator.

### 0.3 Samplers

- **S_trunc(t₀)** (all prior work incl. 2606.13796's annealed schedules): integrate the
  plug-in reverse SDE from T down to t₀ > 0, output x_{t₀}. In the Euler form used in all
  experiments: x ← x + (−½x + ε̂(x,t)/s(t)) dt + √|dt| ξ.
- **S_jump(t̂)** (Tweedie/posterior-mean jump; one-shot optimality for singular data:
  2503.12966): integrate only to moderate t̂, then output

      x̂₀ = (x_{t̂} − s(t̂) ε̂(x_{t̂}, t̂)) / a(t̂).

### 0.4 The normal-variance functional and the recursion

For a probability measure p on R^d, define v(p) := E_p[ dist(X, M)² ] / (d−k), the
per-direction second moment of the normal offset. True data: v = σ². (The ring/sphere
experiments measure this as rvar/offman up to metric constants.)

Self-consuming loop: generation g trains ε̂_g by DSM on pool_g = λ·(fresh real, n samples)
+ (1−λ)·(samples of generation g−1), then samples v_g := v(law of Σ_g's output).

### 0.5 Exact fact: mixtures

**Lemma 0.1 (mixture second moments — exact, no assumptions).** For p = λ p_r + (1−λ) p_s,

    v(p) = λ v(p_r) + (1−λ) v(p_s).

*Proof.* v is the integral of the fixed function dist(·,M)²/(d−k) against p, and
integration is linear in the measure. ∎

Note this is why we work with second moments about M rather than variances: no
mean-alignment assumption (old A1) is needed at this step. A1 is only used later to
interpret v as "variance" (mean offset ≈ 0; measured |rbias| ≤ 0.03 on the ring, §4
tracks the curvature-induced bias on the sphere).

### 0.6 Standing assumptions (each tagged with its evidence)

- **(A0) Scalar reduction.** The per-generation map pool ↦ v(samples) factors through
  w := v(pool): there is a function F_g with v_g = F_g(w_g). This is the reduced-order
  model; it is *not* derived — it is validated by the cross-prediction test (LAW_a↔LAW_b:
  fixed points predicted across two independent experiment types within 1–13%).
- **(A1) Centering.** Mean normal offset is second-order (|bias| ≪ √v). Measured.
- **(A2′) Sampler response (concave law).** F_g(w) = a²(t₀(g)) w + s²(t₀(g)) + e²(w),
  where the per-generation sampling excess e²: [0,∞) → (0,∞) is nondecreasing, concave,
  e²(0) = e₀² > 0. Motivation: §2–§3 (the injection mechanism); measurement: LAW_b
  (local slope c(rate) runs +0.6 at λ=0.75 → ≈0 at λ=0.25 — concave, saturating).
  The affine special case e²(w) = e₀² + c·w is the first-order law of THEORY_NOTE §II.

---

## §1. Theorem II′ — the law of the collapse (nonlinear, concave) [PROVED]

Throughout §1 fix λ ∈ (0,1], write η_g := s²(t₀(g)) + (a²(t₀(g)) − 1) w_g for the
truncation remainder (η_g → 0 for any annealing schedule t₀(g) → 0; for fixed t₀,
η_g → s²(t₀) up to the a² ≈ 1 − t₀ correction, absorbed below). To keep the display
uncluttered we take a² = 1 and carry s²_g := s²(t₀(g)) as the perturbation; the general
case is identical with η_g in place of s²_g.

By Lemma 0.1 and (A0), (A2′):

    w_g = λσ² + (1−λ) v_{g−1},
    v_g = w_g + e²(w_g) + s²_g =: T(v_{g−1}) + s²_g,                     (1.1)

    T(v) := λσ² + (1−λ)v + e²(λσ² + (1−λ)v).

T: [0,∞) → (0,∞) is continuous, nondecreasing, concave (composition of the affine map
v ↦ λσ² + (1−λ)v with the concave nondecreasing w ↦ w + e²(w) is concave nondecreasing),
and T(0) = λσ² + e²(λσ²) > 0.

Let c_∞ := lim_{w→∞} e²′(w) ∈ [0, e²′(0⁺)] (the limit exists: concavity makes e²′
nonincreasing; nonnegativity of the limit from e² nondecreasing; one-sided derivatives
throughout if e² is not differentiable). Define the **critical fraction**

    λ* := c_∞ / (1 + c_∞).

### Theorem II′ (law of the collapse).

**(i) Supercritical λ > λ*: unique plateau.** T has a unique fixed point v_∞ ∈ (0,∞),

    v_∞ solves  v = λσ² + (1−λ)v + e²(λσ² + (1−λ)v),                    (1.2)

and if the crossing is strict (ρ := (1−λ)(1 + e²′(w_∞)) < 1 at w_∞ = λσ² + (1−λ)v_∞),
then for every v_0 ≥ 0 and every truncation schedule with s²_g → 0, the iterates (1.1)
converge: v_g → v_∞. The local rate is ρ.

**(ii) Subcritical λ < λ*: no plateau.** If (1−λ)(1 + c_∞) > 1 then T(v) − v ≥ δ > 0 for
all v ≥ 0 with δ = T(0) ∧ (asymptotic slope gap); consequently v_g ≥ v_0 + gδ − Σ s²
→ ∞. The collapse variance is self-sustaining: no fixed point, no plateau, regardless of
the annealing schedule.

**(iii) Affine corollary (THEORY_NOTE Thm II).** If e²(w) = e₀² + c w, then λ* = c/(1+c),
ρ = (1+c)(1−λ), and

    v_∞ = ((1+c) λσ² + e₀²) / (λ(1+c) − c).                             (1.3)

**(iv) Rate–λ unification.** The measured λ-dependence of the "local c" (LAW_b: +0.6 at
λ=0.75, ≈0 at λ=0.25) is the slope of the single concave function e² evaluated at the
λ-dependent fixed point: local rate ρ(λ) = (1−λ)(1 + e²′(w_∞(λ))). Since w_∞(λ) is
decreasing in λ (smaller λ ⇒ fatter pool at the fixed point) and e²′ is nonincreasing,
e²′(w_∞(λ)) is *increasing* in λ — larger measured local c at larger λ, exactly the
observed pattern. And λ* is governed by c_∞ (the saturated slope), not by e²′(0):
saturation ⇒ c_∞ ≈ 0 ⇒ λ* ≈ 0, matching the measured clean λ* ≈ 0.03.

### Proof.

**Existence and uniqueness of the fixed point (i).** Consider g(v) := T(v) − v, concave
(difference of concave and linear), with g(0) = T(0) > 0. Its one-sided derivative is
g′(v) = (1−λ)(1 + e²′(w(v))) − 1 with w(v) = λσ² + (1−λ)v; g′ is nonincreasing (e²′
nonincreasing, w increasing in v) with limit (1−λ)(1+c_∞) − 1 < 0 when λ > λ*. A concave
function, positive at 0, with eventually negative derivative bounded away from 0, tends
to −∞; hence it has at least one zero. Concavity plus g(0) > 0 gives at most one
*downcrossing* zero: if g(v₁) = g(v₂) = 0 with v₁ < v₂, then by concavity g ≥ 0 on
[0, v₂] ... more precisely g(v) ≥ min(g(0), g(v₂)) · (chord argument) > 0 on (0, v₁) and
g ≤ 0 after the first downcrossing; two zeros would force g ≡ 0 on [v₁, v₂], contradicting
strict negativity of the eventual slope. So the zero v_∞ is unique. Since g > 0 on
[0, v_∞) and g < 0 on (v_∞, ∞):

    T(v) > v  for v < v_∞,     T(v) < v  for v > v_∞.                    (1.4)

**Contraction from above.** For v ≥ v_∞, monotonicity and concavity give
0 ≤ T(v) − T(v_∞) ≤ T′(v_∞⁺)(v − v_∞) = ρ (v − v_∞) with ρ < 1 by hypothesis. So on
[v_∞, ∞), T is a ρ-contraction toward v_∞.

**Convergence with vanishing perturbation (i, continued).** Let v_g = T(v_{g−1}) + s²_g,
s²_g ≥ 0, s²_g → 0.

*Case up (v_{g−1} ≥ v_∞).* Then v_g − v_∞ = (T(v_{g−1}) − v_∞) + s²_g ≤
ρ(v_{g−1} − v_∞) + s²_g. As long as iterates stay ≥ v_∞ the standard perturbed-
contraction lemma applies: for x_g ≤ ρ x_{g−1} + s²_g with ρ ∈ (0,1) and s²_g → 0 one has
x_g → 0 (split the sum Σ_{j≤g} ρ^{g−j}s²_j at g/2; the old terms are killed by ρ^{g/2},
the recent by sup_{j>g/2} s²_j → 0). If some iterate drops below v_∞, pass to Case down.

*Case down (v_{g−1} < v_∞).* Then T(v_{g−1}) ∈ (v_{g−1}, v_∞) by (1.4) and monotonicity
(T(v_{g−1}) < T(v_∞) = v_∞), so v_g = T(v_{g−1}) + s²_g < v_∞ + s²_g. Two sub-cases:
either the sequence stays below v_∞ for all subsequent g — then it is eventually
monotone up to vanishing perturbations: liminf v_g ≥ any accumulation point of the
unperturbed monotone iteration = v_∞ (formally: fix ε > 0; for g large, s²_g ≤ ε(1−ρ̃)
where ρ̃ := sup slope of T on [v_ε, v_∞] < 1 for v_ε := the point where g(v) = ε... —
cleaner: define u_g by u_g = T(u_{g−1}), u_{G₀} = v_{G₀} ≤ v_∞; then u_g ↑ v_∞ by (1.4)
and monotone-boundedness, and v_g ≥ u_g by induction since s²_g ≥ 0 and T monotone;
hence liminf v_g ≥ v_∞) — combined with limsup v_g ≤ v_∞ from the upper barrier
(v_g < v_∞ + s²_g, s²_g → 0), convergence follows; or the sequence crosses above v_∞ at
some g — then Case up takes over, and any later return below re-enters Case down with the
same barriers. In all scenarios limsup and liminf both equal v_∞. ∎(i)

**Local rate.** Linearize (1.1) at v_∞: v_g − v_∞ = ρ(v_{g−1} − v_∞) + o(|v_{g−1} − v_∞|)
+ s²_g, with ρ = T′(v_∞) = (1−λ)(1 + e²′(w_∞)). ∎

**(ii) Subcritical divergence.** If (1−λ)(1 + c_∞) > 1 then g′(v) = (1−λ)(1+e²′) − 1 ≥
(1−λ)(1+c_∞) − 1 =: δ₁ > 0 for all v (e²′ ≥ c_∞ everywhere by concavity — the derivative
decreases *to* its limit). So g is nondecreasing with g(0) = T(0) > 0, giving
T(v) − v ≥ T(0) ∧ ... in fact simply g(v) ≥ g(0) > 0 for all v. Then
v_g ≥ v_{g−1} + g(v_{g−1}) ≥ v_{g−1} + T(0) (the s²_g ≥ 0 only helps), so v_g ≥ v_0 +
g·T(0) → ∞, and in fact once v is large the increments approach slope δ₁·v — at least
linear growth, no plateau for any schedule. ∎(ii)

**(iii)** Substitute e²(w) = e₀² + cw: T(v) = (1+c)(λσ² + (1−λ)v) + e₀², an affine map
with slope ρ = (1+c)(1−λ). ρ < 1 ⟺ λ > c/(1+c). Fixed point: solve v = ρv + (1+c)λσ² +
e₀², i.e. v_∞ = ((1+c)λσ² + e₀²)/(1 − (1+c)(1−λ)) and 1 − (1+c)(1−λ) = λ(1+c) − c. ∎(iii)

**(iv)** is a restatement of the formulas in (i) plus monotonicity of w_∞(λ): from (1.2),
implicit differentiation of v_∞(λ) (or the affine case explicitly) shows v_∞ decreasing
in λ, hence w_∞ = λσ² + (1−λ)v_∞ decreasing in λ whenever v_∞ ≫ σ² (the collapse regime).
∎

### Remarks (referee anticipation).

1. **What is proved vs what is assumed.** Given (A0) + (A2′), everything in Theorem II′
   is unconditional elementary analysis. The scientific content of the paper is that
   (A2′) — a one-dimensional, concave response law — *suffices* to reproduce every
   measured recursion phenomenon: geometric approach (LAW_b trajectories), fixed-point
   values (cross-prediction 1–13%), the λ-dependent local rates (iv), the tiny λ*
   (c_∞ ≈ 0), and the annealed transient (s²_g enters additively — the earlier "−0.5
   slope" artifact is the polynomial s²(g) = 0.5/(1+g)² transient superposed on the
   geometry, resolved by fixed-t₀ runs).
2. **Why c is small: a MEASURED two-channel near-cancellation [poolwidth_probe.py, 2026-07-03].**
   The response e²(w) = v_output(pool=w) − w has two competing, opposite-signed,
   comparable-magnitude channels:
   - **Deficit response (deterministic, NEGATIVE):** dΦ_det/dw − 1. A fatter pool has a
     smaller peak required slope κ_max(w) = 1/(2√w), so the deficit closes and the
     excess falls. Computed (deficit_floor_law): c_deficit ≈ −0.68 at w = σ².
   - **Injection response (stochastic, POSITIVE):** a fatter pool damps the injected
     residual field less (weaker contraction 1/γ²_w), retaining more. c_inj ≈ +0.73.
   The NET is the measured law slope: poolwidth_probe gives c_total = dv/dw − 1 =
   {0.00, 0.05, 0.13, 0.10, 0.05} across w = {0.0009, 0.0025, 0.0049, 0.010, 0.0169} —
   small, positive, and CONCAVE (peaks near w ≈ 0.005), matching LAW_b's c ≈ 0.03–0.06
   and its non-constant concave character. So c ≈ 0.05 is the residual of two O(0.7)
   mechanisms, not a small primary effect — and λ* = c_∞/(1+c_∞) ≈ 0.05 follows from the
   large-w tail where c decreases. [Challenge #10 RESOLVED: my earlier retention/smoothing
   hypothesis was the wrong pair; the correct pair is deficit (deterministic) vs injection
   (stochastic), both now measured.] (A2′) remains a measured law with this two-channel
   anatomy; no first-principles derivation of the full shape is claimed.
3. **Truncation enters only through the additive s²_g** — this is the precise sense in
   which "annealing targets the wrong term" (2606.13796 Case 2): the schedule can kill
   s²_g but never e², and Theorem II′(i) says the plateau is then (1.2)-determined by e²
   alone. The Cambridge fix is recovered as the special case e² ≡ 0 (their Case 1
   assumptions imply the estimation term is schedule-dominated), where v_∞ = σ²: their
   Theorem 4.1 dynamics are the e² → 0 limit of (1.1).

---

## §2. Propagation — how score error becomes normal variance [COMPLETE: Lemmas 2.1–2.2 and Prop 2.3 PROVED; Prop 2.4 proved in block model (R′); Prop 2.5 verified zero-fit]

This section is the shared infrastructure for Theorems I and III. Everything is in the
normal coordinate of §0.2.

### 2.1 The variance ODE

For an affine-in-x estimator ε̂(x,t) = κ̂(t)x + b(t) (the true predictor is κ(t)x), the
normal variance V(t) of the sample ensemble under the reverse Euler scheme of §0.3 obeys

    −dV/dt = (1 − 2 κ̂(t)/s(t)) V + 1                                    (2.1)

(derivation: Lemma 2.1 below; ε̂/s enters the drift, code form
x + (−½x + ε̂/s)dt + √|dt| ξ).

**Check (exactness).** With the true slope κ̂ = κ = s/γ²: κ/s = 1/γ², and V(t) = γ²(t)
solves (2.1): −d(γ²)/dt = −a²(1−σ²) and (1 − 2/γ²)γ² + 1 = γ² − 1 = −a²(1−σ²). ✓ The
plug-in reverse SDE with the exact score transports the exact marginals — (2.1) is the
right bookkeeping object.

### 2.2 The two-channel error decomposition [MEASURED 2026-07-03 — rprime_probe suite]

Write the trained estimator as ε̂(x,t) = κ̂(t)x + b(t) + δ(x,t) (Lemma 2.2's
decomposition). The R′-probe (rprime_probe.py / rprime_probe2.py, ring R = 2.5,
σ = 0.05, validated protocol — the same net reproduces the LAW_a-era floors exactly:
std excess 0.025/0.022/0.016 at t₀ = 0.02/0.005/0.001) measured both channels directly
against the exact quadrature score:

- **Slope-deficit channel (dominant at practical t₀).** The fitted slope tracks the true
  slope faithfully at benign times (κ̂/κ* = 0.95–0.97 for t ≥ 0.1) and then SATURATES at
  a ceiling κ̄ ≈ 3.8–4.0 ≈ κ*(0.1) while κ* continues to 9–10 (ratio 0.61 at t = 0.02,
  0.42 at 0.005) — an *optimization/pipeline ceiling*, n-independent. Naive analysis of
  (2.1) says a capped slope still contracts at diverging rate κ̄/s, so the equilibrium
  V_eq = s/(2κ̄ − s) → 0 and there is "no floor" (the first form of challenge #7). That
  neglects the **relaxation lag**: the number of contraction e-folds remaining from t
  down to t₀ is ∫ 2κ̄/s dt ≈ 4κ̄(√t − √t₀) — *finite* — so V cannot track its shrinking
  equilibrium and freezes at a quasi-floor. **Zero-free-parameter verification
  (Prop 2.5):** integrating (2.1) with the measured κ̂(t) reproduces 69–75% of the
  measured sampling floors at all three t₀ (0.0353/0.0206/0.0140 predicted vs
  0.0472/0.0297/0.0198 measured), with the exact-score control reproducing γ²(t₀) to
  all digits. The deficit-with-lag channel is the dominant floor mechanism.
- **Residual-field channel (injection; the schedule-proof guarantee).** The orthogonal
  residual has risk r²_res(t) ≈ 0.02–0.05, roughly FLAT over the whole time range
  (measured; = the r̂² ≈ 0.03 previously inferred from T1's jump excess — independent
  confirmation). It enters the drift as δ/s and can only inject (Lemma 2.2), with
  measured path-coherence ν ≈ 0.72–0.78. Its contribution to the floor here is
  subdominant (~30% together with the bias field b: the probe shows b(0, t=0.02) ≈
  +0.25 in ε-units, an angularly-varying radial bias). Prop 2.4 lower-bounds the floor
  through this channel alone — weaker than the measured total, but it is the component
  that survives ALL schedules and pipelines with ρ₀ > 0, hence the impossibility
  statement runs through it.
- **Sharpening (the signed third face).** Where the *fitted* part over-contracts
  relative to the data (posterior-mean geometry, Prop 2.3), output variance falls below
  σ² — the thin channel (MNIST axes with σ_ax ~ s(t̂)). Deficit fattens, sharpening
  thins, residual injects: all three faces are measured, and the *sign* of the needed
  correction is regime-dependent — which is the proof that any correct matcher must be
  bidirectional.

**Assumption R (band risk), restated on measured footing.** r²_total(t) = deficit² · γ²
+ r²_res(t) rises into the band (measured: 0.036 at t = 0.1 → 0.271 at t = 0.005; the
decomposition identity (κ*−κ̂)²γ² + r²_res = r²_total checks numerically: 0.226 + 0.035
= 0.261 ≈ 0.271 at t = 0.005). n-independence: LAW_a (floors flat 1k→16k).

**Assumption R (band risk).** There exist t* > 0 and ρ₀ > 0, independent of n, with
r²(t) ≥ ρ₀ for t ≤ t*, for every estimator produced by the (fixed) training pipeline on
tube data. [Motivated by the κ ≈ 1/s divergence + finite capacity; validated by LAW_a
n-flatness. Theorem I is stated conditionally on R — this is the honest fallback of
challenge log #4, now with the right mechanism attached.]

**Sign summary (unifies the record):** total normal-variance error =
(sharpening, ≤ 0, from slope deficit & posterior-mean contraction) + (injection, ≥ 0,
from stochastic field error, 1/s²-weighted). Truncated SDE on thin tubes: injection
dominates ⇒ fat floors (all LAW arms). Jump at σ_ax ~ s(t̂): sharpening dominates on thin
axes ⇒ MNIST thin_ratio 0.62 (T4). Jump ablation on the ring: injection through the
recursion ⇒ fat plateau e²/λ (challenge #2's resolution). **Corollary: any correct
matcher must be bidirectional** — the sign of the required correction is regime-
dependent. [To be formalized as Prop 2.3 + used in §4.]

### Lemma 2.1 (variance ODE). [PROVED]

Under the Euler reverse scheme x′ = x + (−½x + ε̂(x,t)/s(t))(−dt) + √dt ξ, dt > 0,
ξ ~ N(0,1) independent, with affine estimator ε̂(x,t) = κ̂(t)x + b(t), the variance V(t)
of the sample ensemble satisfies, in the dt → 0 limit,

    −dV/dt = (1 − 2κ̂(t)/s(t)) V + 1.                                     (2.1)

*Proof.* x′ = (1 + (½ − κ̂/s)dt) x − (b/s)dt + √dt ξ. The affine-in-x map plus
independent noise gives Var(x′) = (1 + (½ − κ̂/s)dt)² V + dt = V + (1 − 2κ̂/s)V dt + dt
+ O(dt²). Rearranging and letting dt → 0 gives (2.1); b affects only the mean. ∎

(Exactness check with κ̂ = κ: §2.1. The bias/mean ODE, needed for the sphere curvature
discussion, is −dμ/dt = (½ − κ̂/s)μ − b/s — first-order decoupled.)

**Terminology (two distinct "injections" — do not conflate in the manuscript).**
(1) **Sampler-noise injection**: the +1 forcing in (2.1), i.e. the reverse-SDE's own
Brownian term √|dt| ξ. This is a property of the *sampler class*, present even for a
perfect score; it is what the two-horned ablation (§3) switches off, and it is
load-bearing for the floor's existence (OFF ⇒ V → 0, point collapse; ON ⇒ Φ_det).
(2) **Residual-field injection**: the accumulated kicks δ(x_τ,τ)/s(τ) of the estimator's
non-affine residual (Lemma 2.2, Prop 2.4) — a property of the *estimator*, entering on
top of (2.1) and lower-bounded through (R′). Φ_det (Theorem I(a)) is the floor of the
capped-slope sampler WITH (1) but WITHOUT (2); the measured floor adds (2)'s ~30%
(single-pass). Where a sentence says only "injection," it means (2) in §2 and the +1
of (1) in the §3 ablation discussion.

### Lemma 2.2 (no first-order thinning from the residual — the orthogonality shield). [PROVED, unconditional]

Let ε̂(·, t) be ANY estimator (no optimality or projection hypothesis). Define
(κ̂(t), b(t)) as the affine L²(p_t)-fit *of the estimator itself*,
(κ̂, b) = argmin E_{p_t}(ε̂(x) − κx − b)², and δ := ε̂ − κ̂x − b the residual. Then,
unconditionally,

    E_{p_t}[δ] = 0   and   E_{p_t}[x δ] = 0

(normal equations of the affine fit). Consequently the residual has *no first-order
effect* on the variance flow (2.1) at the forward marginals: the first variation of V
under the perturbed drift (κ̂x + b + η·δ)/s is (2/s)·E_{p_t}[x δ]·dt = 0 at η = 0.
The leading effect of δ is second order — the accumulated displacement
u(t) = ∫ δ(x_τ, τ)/s(τ) dτ enters as E[u²] ≥ 0: **injection, never first-order
contraction.** (Measure caveat, stated precisely: orthogonality holds under p_t; the
flow ensemble p̂_t deviates from p_t along the reverse path, so the cross term
E_{p̂_t}[xδ] is O(‖p̂_t − p_t‖ · r(t)) — second order in the total perturbation,
consistent with the claim, and absorbed into the constants of Prop 2.4.)

*Why this lemma matters.* Every estimator splits exactly into (slope + intercept) +
(orthogonal field residual). Thinning/sharpening can only enter through the affine part
(κ̂ ≠ κ — a genuine slope deficit); the residual field can only inject variance. This
closes the referee objection that "the estimation error could just as well thin the
tube" (which sank the naive floor argument, challenge #7) — and it does so with NO
hypothesis on the training. [SUPERSEDED CLAIM, kept for the record: an earlier version
of this paragraph asserted "the assumptions of the floor live entirely in (R′)" and
"δ ≡ 0 ⟹ the floor tracks s/(2L) → 0, no floor at all." Both are FALSE post-Theorem
I(a): the equilibrium-tracking argument neglects the relaxation lag (Lemma 3.0's finite
band), and a pure affine capped-slope estimator under the stochastic sampler floors at
Φ_det(κ̄) > σ² with δ ≡ 0. The floor's assumption surface is κ̄ < ∞ (§3); (R′) governs
only the residual channel's ~30% single-pass addition. What survives of the abductive
point: the *n-independence* of the measured floor localizes correctly — κ̄ and r² are
pipeline/optimization budgets, not information limits.]

**Capacity note (recorded, important for the paper's framing).** At σ = 0.05 the maximal
required slope is κ_max ≈ 1/(2σ) = 10 — trivially representable. So the measured floors
(LAW_a) are NOT a 1-d slope-capacity effect; r²(t) is the residual of fitting the score
*field* jointly over (x, t) — a smoothness/optimization budget across the whole tube and
schedule, which is exactly why it is n-independent and why Assumption R (band risk) is an
assumption about the pipeline, not about information. The 1-d κ-blowup enters the story
as σ → 0 (Case-2 limit): κ_max → ∞ forces either the slope deficit (sharpening) or a
diverging weight budget — the dichotomy of Theorem I(c).

### Prop 2.3 (Tweedie sharpening — exact per-axis formula). [PROVED]

In the Gaussian normal coordinate with axis variance σ²_ax, the jump output with the
*exact* predictor is x̂₀ = (aσ²_ax/γ²_ax) x_{t̂}, i.e. exactly the posterior mean, with

    Var(x̂₀) = a² σ⁴_ax / γ²_ax(t̂),      sharpening factor  Var(x̂₀)/σ²_ax
             = a² σ²_ax / (a² σ²_ax + s²(t̂))  < 1.

With a trained predictor ε̂ = ε* − δ̃: x̂₀ = (aσ²_ax/γ²_ax) x + (s/a) δ̃, so

    v_jump,ax = a²σ⁴_ax/γ²_ax + (s²/a²) E[δ̃²] + 2(s/a)·Cov(m, δ̃),      (2.2)
    |cross| ≤ 2(s/a)·(a²σ⁴_ax/γ²_ax)^{1/2} (E δ̃²)^{1/2}   (Cauchy–Schwarz),

plus curvature O(s⁴/τ²) outside the flat idealization (posterior-mean chord bias).

*Proof.* x̂₀ = (x − sε*(x))/a = x(1 − s²/γ²)/a = x·(a²σ²/γ²)/a = (aσ²/γ²)x, and
Var = (aσ²/γ²)²·γ² = a²σ⁴/γ². Tweedie identifies this with E[X₀|X_{t̂}]. The error term
is the definition of the jump map applied to ε* − δ̃. ∎

**Consequences.**
(a) *Strongly singular axes* (σ_ax ≪ s(t̂)): factor ≈ a²σ²_ax/s² ≈ 0 — the jump lands
essentially ON the manifold; the matcher must top up to σ² (bidirectionality, thin side).
The endpoint after top-up is σ²_ax + (s²/a²)r²(t̂): the network error enters *multiplied*
by s²(t̂) — the class-separation mechanism of Theorem I(b).
(b) *Mildly singular axes* (σ_ax ~ s(t̂)): factor strictly inside (0,1) and
axis-dependent — anisotropic thinning. A scalar (total-energy) matcher cannot be correct
on all axes simultaneously ⇒ per-axis matching required. [Direction and regime boundary
confirmed: MN arm |thin−1| ≤ 0.012 with the per-axis matcher vs 0.62 scalar. The
*numerical* 0.62 is a recursion+scalar-matcher compound, not this one-step factor —
challenge log #9.]
(c) *Fat axes* (σ_ax ≫ s): factor ≈ 1 — no sharpening; injection dominates ⇒ fat side of
the matcher.

### Assumption R′ and Prop 2.4 (injection lower bound) [PROVED in the block model]

**(R′)** Along the reverse path, the residual field δ has (i) band risk: E[δ²](t) ≥ ρ₀
for t ≤ t* (Assumption R); (ii) block structure: [t₀, t*] partitions into intervals
("blocks") on which the kicks δ(x_τ, τ)/s(τ) accumulated by a path are coherent within a
block (correlation time ≥ ν t at time t, self-similar) and independent across blocks;
(iii) no systematic over-contraction: κ̂(t) ≤ κ(t) (justified: the affine part of the
L²(p_t) projection is exactly κ, deficits only from capacity). [The modeling content is
(ii); challenge #8.]

**Prop 2.4 (the floor, with its shape).** Under Lemma 2.2 + (R′), for every truncation
time t₀ ≤ t* and every schedule,

    V(t₀) ≥ γ²(t₀) + Φ₀(t₀),      Φ₀(t₀) ≥ c ν ρ₀ (σ² + t₀),            (2.3)

with c an absolute constant. In particular: Φ₀ is **decreasing in t₀ until t₀ ~ σ², then
saturates at c ν ρ₀ σ² — annealing t₀ → 0 cannot go below a fixed positive floor**, and
the floor is *relative*: V(t₀)/σ² ≥ 1 + cνρ₀ uniformly in the schedule. All constants
are n-independent (ρ₀ is a pipeline/capacity constant, Assumption R).

*Proof.* Decompose the output deviation into per-block contributions. A block at time t
has length ℓ(t) = νt; within it the path accumulates a coherent displacement of variance
≥ ρ₀ ℓ(t)²/s²(t) (coherence (ii); risk (i)). A displacement present at time t is damped,
by the linearized flow with slope κ̂ ≤ κ (iii), by at most the exact-score factor

    D(t, t₀) = exp( −∫_{t₀}^{t} (1/γ²(τ) − ½) dτ ) ≤ (σ² + t₀)/(σ² + t) · e^{t/2},

using γ²(τ) = a²σ² + s² = σ² + τ(1−σ²) + O(τ²) so ∫ dτ/γ² = log((σ²+t)/(σ²+t₀)) + O(t).
Independence across blocks (ii) makes the variances add; converting the block sum to an
integral with block density dt/ℓ(t):

    E[u²](t₀) ≥ Σ_blocks ρ₀ (ℓ²/s²) D² ≥ c ∫_{t₀}^{t*} ρ₀ · (ν²t²/s²(t)) ·
                ((σ²+t₀)/(σ²+t))² · dt/(νt)
              = c ν ρ₀ ∫_{t₀}^{t*} (t/s²(t)) ((σ²+t₀)/(σ²+t))² dt
              ≥ c′ ν ρ₀ (σ²+t₀)² [ (σ²+t₀)^{−1} − (σ²+t*)^{−1} ]      (t/s² ≥ c on [0,T])
              ≥ c″ ν ρ₀ (σ² + t₀)                                       (t* ≫ σ²).

The cross term 2E[x u] vanishes at first order by Lemma 2.2 (residual ⊥ x in L²(p_t)),
and the flow part is ≥ γ²(t₀) by (iii) (under-contraction can only add). Summing gives
(2.3). ∎

### Prop 2.5 (zero-free-parameter floor postdiction — the deficit channel). [VERIFIED 2026-07-03]

Integrating the variance ODE (2.1) with the *measured* fitted-slope profile κ̂(t)
(rprime_probe: ratio 0.95 → 0.42 over t = 0.5 → 0.005, ceiling κ̄ ≈ 3.9) from T = 8 down
to t₀ reproduces 69–75% of the same net's measured sampling floors with NO free
parameters: predicted V(t₀) = 0.0353 / 0.0206 / 0.0140 vs measured 0.0472 / 0.0297 /
0.0198 at t₀ = 0.02 / 0.005 / 0.001. Control: the same integration with the exact slope
κ*(t) returns γ²(t₀) to all displayed digits at every t₀ — the integrator and the ODE
are exact. The ~30% remainder is the residual-field injection + bias channel (Prop 2.4's
subject). Together: the floor mechanism is empirically closed — deficit-with-lag
dominant, injection/bias the remainder and the schedule-proof guarantee.

**Confrontation with LAW_a (the shape test, passed — with honest error bars).** (2.3)
predicts the one-shot excess over σ² + s²(t₀) to be ≈ νρ₀(σ² + t₀): *decreasing* in t₀
while t₀ > σ², then *saturating* below t₀ ≈ σ². Measured (LAW_a): excess monotone
decreasing down to t₀ = 0.001 and flattening at 0.017–0.020, with σ² = 0.0025 — the knee
sits where the theory puts it, and n-flatness across 1k→16k ✓ (no n anywhere in (2.3)).
Calibration is order-one but not constant: fitting νρ₀ at t₀ = 0.005 gives ≈ 3, at
t₀ = 0.001 gives ≈ 5 — a ±2× drift across the range. The drift direction is exactly what
a *non-constant* r²(t) rising toward the band edge produces (constant-ρ₀ is the
conservative floor; the true injection grows as t ↓). So the theorem's content — shape,
knee location, positivity, schedule-independence, n-independence — is confirmed; the
single-constant calibration is effective to within a factor ~2, and the paper should
present νρ₀ as a band (≈ 3–5), organizing THEORY_NOTE's "≈ 8× σ² saturation" as
1 + cνρ₀.

---

## §3. Theorem I — the floor and the class separation [PROVED — CLOSED FORM in the singular limit; unconditional variant included]

**Restructured 2026-07-03 (deficit_floor_law.py); analytic closed forms 2026-07-04
(floor_theory_upgrade.py, closed_form_check.py).** The impossibility no longer rests on
the stochastic assumption (R′). It rests on a single property that every finite neural
network has automatically: a **bounded normal slope**, κ̂(t) ≤ κ̄ < ∞ (finite Lipschitz
constant in the normal coordinate). (R) and (R′) now govern only the subdominant
injection *correction*, not the floor's existence. As of 2026-07-04 the floor law is
exact: a σ-free limit ODE (Lemma 3.2) with closed-form solutions (Lemma 3.3), in TWO
classes — unconditional (κ̂ ≤ κ̄ only, overshoot allowed) and no-overshoot (κ̂ ≤ κ ∧ κ̄,
the measured profile) — whose asymptotic constants differ by only 5.5%.

### Lemma 3.0 — the required slope is bounded and non-monotone.

The true normal slope κ(t) = s(t)/γ²(t) (§0.2) rises from 0, peaks at
**κ_max = 1/(2σ)** at t ≈ σ² (where s ≈ σ), then FALLS back to 0 as t → 0 (γ² → σ²,
s → 0). [Verified: peak at t = 0.00251 = σ², κ_max = 10.01 = 1/(2·0.05).] Hence for any
ceiling κ̄ < κ_max the set {t : κ(t) > κ̄} is a *finite band* [t_lo, t_hi]; below t_lo
the exact slope is again representable. This non-monotonicity is the crux the naive
Lipschitz argument (challenge #7) missed.

### Lemma 3.1 — monotone comparison for the variance ODE.

Let V_κ̂ solve the variance ODE (2.1), −dV/dt = (1 − 2κ̂(t)/s(t))V + 1, from V(T)=1 down
to t₀. If κ̂_A(t) ≤ κ̂_B(t) for all t, then V_{κ̂_A}(t₀) ≥ V_{κ̂_B}(t₀).

*Proof.* In the time-to-go variable u = T − t, (2.1) is the linear scalar ODE
dV/du = p(u)V + 1 with p = 1 − 2κ̂/s and common positive forcing q ≡ 1, common
V(0) = 1. κ̂_A ≤ κ̂_B ⟹ p_A ≥ p_B pointwise; the comparison principle for linear scalar
ODEs with equal forcing and initial data gives V_A ≥ V_B for all u. ∎

### Lemma 3.2 — the singular-limit ODE (universality is exact). [PROVED 2026-07-04]

Rescale t = σ²u, V = σ²W, and write the slope profile as κ̂(t) = (1/σ) m(u) with

    m(u) = ρ/2                                (unconditional envelope, κ̂ ≡ κ̄)
    m(u) = min( √u/(1+u), ρ/2 )               (no-overshoot envelope, κ̂ = κ ∧ κ̄),

ρ := 2σκ̄ = κ̄/κ_max. Using s²(σ²u) = σ²u·(1+O(σ²u)) and γ² = σ²(1+u)(1+O(σ²u)),
equation (2.1) becomes −dW/du = 1 − (2m(u)/√u)W + O(σ²)·W, i.e. as σ → 0 at fixed ρ,

    −dW/du = 1 − (2 m(u)/√u) W                                          (3.1L)

— a **σ-free ODE depending only on ρ**. The uncapped equation (m = √u/(1+u)) has the
EXACT particular solution W = 1+u (the rescaled true marginal γ²/σ²) and general
solution W = (1+u) + A(1+u)²; downward integration contracts deviations as
((1+u)/(1+u′))², so the solution entering the band is the slave value 1+u₊ with error
O(σ²/ρ²) from (i) the coefficient expansion over the band range u ≤ u₊ = O(1/ρ²) and
(ii) the boundary layer at t ≳ 1, suppressed quadratically by the slave contraction.
Hence Φ(κ̄, σ)/σ² = g_class(ρ) + O(σ²/ρ²), with g_class determined by (3.1L) alone:
**universality in ρ is a theorem, not a numerical observation.** [Convergence verified:
full ODE at σ = 0.10/0.05/0.02 vs (3.1L): e.g. ρ = 0.39: 3.953/3.812/3.775 → 3.769;
ρ = 0.20: 17.0/14.1/13.57 → 13.47 — approach from above at rate consistent with
O(σ²/ρ²); slave invariance checked to 3×10⁻¹⁵. floor_theory_upgrade.py.] ∎

### Lemma 3.3 — closed forms for the floor. [PROVED 2026-07-04; verified to 2×10⁻⁵]

**(a) Unconditional class (κ̂ ≤ κ̄ one-sided; overshoot allowed).** With m ≡ ρ/2, (3.1L)
integrates exactly: d/du[W e^{−2ρ√u}] = −e^{−2ρ√u}, and for any data of sub-quadratic
growth at u = ∞,

    g_const(ρ) = W(0⁺) = ∫₀^∞ e^{−2ρ√u} du = 1/(2ρ²)      **EXACT, every ρ > 0.**

In original variables: **Φ_uncond → σ²/(2ρ²) = 1/(8κ̄²)** — a closed-form,
σ-independent floor from the Lipschitz constant alone. g_const > 1 ⟺ ρ < 1/√2: any
network with slope ceiling κ̄ < κ_max/√2 = 1/(2√2σ) is floored above σ² with NO further
assumption.

**(b) No-overshoot class (κ̂ ≤ κ ∧ κ̄; the measured profile, rprime_probe).** The band
edges solve √u/(1+u) = ρ/2, i.e. ρx² − 2x + ρ = 0 in x = √u:

    x∓ = (1 ∓ q)/ρ,   q := √(1−ρ²)   (band nonempty ⟺ ρ < 1);  u∓ = x∓².

Above the band the solution is the exact slave W = 1+u. Across the band, the (a)
integrating factor with antiderivative F(u) = −(2ρ√u+1)e^{−2ρ√u}/(2ρ²) gives, using
2ρx∓ = 2(1∓q) and 2ρ(x₊−x₋) = 4q,

    W(u₋) = (1+u₊) e^{−4q} + [ (3−2q) − (3+2q) e^{−4q} ] / (2ρ²).

Below the band the equation is uncapped with exact general solution (1+u) + A(1+u)²,
so the floor is reached by pure algebra:

    g(ρ) = 1 + [ W(u₋) − (1+u₋) ] / (1+u₋)².                            (3.2C)

Properties (all from (3.2C)): g(ρ) > 1 for every ρ ∈ (0,1); g(1⁻) = 1 (band vanishes);
and as ρ → 0,

    g(ρ) = C₀/ρ² + O(1),   **C₀ = (1+3e^{−4})/2 = 0.527473…**,

so **Φ_no-ovr → C₀σ²/ρ² = (1+3e^{−4})/8 · κ̄^{−2} = 0.131868/κ̄²** — again σ-independent.
[closed_form_check.py: (3.2C) and (a) match direct integration of (3.1L) to ≤ 2.4×10⁻⁵
across ρ ∈ [0.10, 1.00] — pure Euler residual. The two asymptotic constants, 1/8 = 0.125
and 0.131868, differ by 5.5%: **the overshoot question is quantitatively negligible**,
which retires open item 4.] ∎

### Theorem I(a) — the deterministic collapse floor (assumption-light).

**Two classes, two floors — both closed-form in the singular limit.** [The 2026-07-03
version stated the bound via the constant-κ̄ comparison while quoting min-mode numbers —
a statement/numbers mismatch, caught and repaired 2026-07-04; challenge #13.]

**(U) Unconditional.** Let the estimator satisfy only κ̂(t) ≤ κ̄ < ∞ (one-sided: no
lower bound, arbitrary overshoot allowed — every finite network qualifies). Then for
every t₀ > 0 and every schedule,

    V(t₀) ≥ V_{κ̂≡κ̄}(t₀)  →(σ→0)  σ²·(1/(2ρ²))·(1+o(1)) = (1/(8κ̄²))(1+o(1)),  (3.0U)

by Lemma 3.1 (one-sided comparison suffices: smaller κ̂ only raises V) and Lemma 3.3(a).
The floor exceeds σ² whenever ρ < 1/√2, i.e. κ̄ < 1/(2√2 σ).

**(N) No-overshoot (the measured class).** If additionally κ̂(t) ≤ κ(t) (never
over-contract; measured: rprime_probe κ̂/κ* = 0.95–0.97 at benign t, saturation *below*
κ* in the band, never above; population-DSM motivation: the affine L²(p_t)-projection
of the true target is exactly κ), then the tighter comparison κ̂ ≤ min(κ, κ̄) applies:

    V(t₀) ≥ Φ_det(κ̄) := σ² g(2σκ̄) (1+o(1)),   g the closed form (3.2C),      (3.0N)

with:
 (i)  **Monotone, exhaustive:** g strictly decreasing on (0,1), g(ρ) > 1 for EVERY
      ρ < 1, g → 1 only as ρ → 1⁻ (κ̄ → κ_max): within this class no finite network
      reaches σ². [Finite-σ values at σ = 0.05: Φ_det/σ² = 14.1, 6.28, 3.82, 2.44,
      1.43, 1.00 at κ̄ = 2, 3, 3.9, 5, 7, 10 — exceeding the σ→0 closed form by 1–8%,
      the O(σ²/ρ²) correction of Lemma 3.2.]
 (ii) **Universal:** Φ_det/σ² = g(ρ), ρ = 2σκ̄ — now a THEOREM (Lemma 3.2), with the
      numeric σ-collapse (1–8% across σ ∈ {0.02, 0.05, 0.10}) as its convergence check.
 (iii)**Strongly-singular asymptotic — exact constants:** as ρ → 0,
      Φ_uncond → 1/(8κ̄²) = 0.125/κ̄²  and  Φ_no-ovr → (1+3e^{−4})/8·κ̄^{−2}
      = 0.131868/κ̄² — σ-INDEPENDENT: the off-manifold floor is set by the network's
      slope ceiling alone. The two constants differ by 5.5%, so the class distinction
      is quantitatively minor. [The earlier "0.141 ± 0.003" (σ = 0.05, κ̄ ∈ {1.5..3})
      is the finite-σ effective value of the exact 0.131868; the +5–8% is the measured
      O(σ²/ρ²) correction, consistent by direction and size.]

*Proof.* Comparison (Lemma 3.1) against the class envelope; the envelope's floor is
Lemma 3.3(a) for (U) and Lemma 3.3(b) for (N); Lemma 3.2 transfers the limit-ODE values
to the finite-σ ODE with O(σ²/ρ²) error. Mechanism (Lemma 3.0): the finite band
[t_lo, t_hi] where κ̄ < κ(t) freezes in variance, and below t_lo the remaining
relaxation e-folds 2log(1+ρ²/4) ≤ ρ²/2 cannot re-thin it. The ODE is exact for the
affine-in-normal score (Lemma 2.1); by Lemma 2.2 the non-affine residual only *adds*
variance, so both bounds hold for the full estimator. ∎

**Recursion version.** Under (A0) the per-generation floor at pool variance w is
Φ_det(κ̄; w) ≥ Φ_det(κ̄) (a fatter pool only relaxes the requirement), so every
generation's output is ≥ Φ_det(κ̄) independent of v_{g−1}; hence liminf_g v_g ≥
Φ_det(κ̄) > σ². Annealing t₀(g) → 0 changes nothing — the floor is *below* any t₀ and
is the same limit. **This is the answer to 2606.13796's open Case 2: annealed
truncation cannot converge to the data distribution because the barrier is the finite
slope of the learned score, which no schedule touches.** ∎

**What the network measurement gives (Prop 2.5, now a corollary).** rprime_probe
measured κ̄ ≈ 3.9 for the standard pipeline; (3.0N) then predicts Φ_det ≈ 3.8σ² ≈ 0.0095
(finite-σ value; closed form (3.2C) gives g(0.39) = 3.77 in the σ→0 limit),
matching the measured *single-pass* t₀ → 0 sampling floor. The floor is *predicted from
one measured network property, deterministically, with no fit.*

### Empirical confirmation of Theorem I on the recursion (T5–T7, 2026-07-04).

**Head-to-head vs 2606.13796's remedy (head_to_head.py, 5 seeds, G=20, λ=½).** Arm A =
annealed truncation t₀: 0.05→1e-4, no matcher (Cambridge's method). Arms B/C = our fix
(std+match, jump+match). Result: **Arm A saturates at 9.84σ² ± 1.7σ² (rvar 0.0246 ± 0.0041)** and
never reaches the tube; **Arms B and C hold 1.01σ² ± 0.02σ², dead flat for 20
generations** on both samplers. This is Theorem I(a)'s recursion floor made visible: the
Cambridge line is stuck ≫ σ², the anchored line is pinned at σ². [Items 2, 3: closed.]

**Two-horned trap — the injection ablation (inject_ablation.py + verify_nonoise.py).**
Fixed t₀=1e-4, no matcher, reverse-SDE with the stochastic injection term ON vs ZEROED,
3 seeds each:
 • **Stochastic (injection ON):** saturates at **10.8–13.2σ²** — the fat floor of
   Theorem I(a) (the autonomous fixed point, slightly above Arm A's not-fully-converged
   annealed endpoint).
 • **Deterministic (injection OFF):** holds NO floor — it collapses to a **single point**
   (2D coordinate std → 4·10⁻⁴ by gen 2, verified NaN-free), i.e. V → 0 *through* σ².
This is exactly the structure of the variance ODE (2.1): the **+1 injection is the sole
term that makes the floor nonzero**; remove it and the capped-drift contraction runs to 0
(total mode collapse). Two failure modes — stochastic overshoots to a fat floor,
deterministic under-collapses to a point — and **neither reaches σ²**; the matcher fixes
both. [Corrects the earlier "Φ_det is deficit-only, omits injection" statement: Φ_det is
the *with-injection* reverse-SDE floor. Challenge #12.]

**Real-data step — pixel MNIST (pixel_mnist_recursion.py, 8×8 pixels, G=8, 3 seeds).**
Off-manifold distance to a held-out real test set (true-data baseline 1.34): the unfixed
recursion drifts to **3.06** (2.3× baseline, collapse); the fix holds **1.66** (1.24×
baseline, near the real manifold), closing ~81% of the gap — with a mild residual creep (a
local kNN-PCA chart only approximates a curved manifold, vs the exact ring). [Item 4:
closed on real pixels.]

**Honest quantitative gap.** The measured *recursion* stochastic floor (≈11σ²) sits 2.4×
above Φ_det's recursion fixed point (v = Φ_det(½σ² + ½v) = 4.54σ² at t₀=1e-4, κ̄=3.9).
Theorem I(a)'s recursion version claims only a *lower* bound (liminf v_g ≥ Φ_det), which
11σ² respects; the 2.4× excess is per-generation compounding of score degradation (a net
trained on self-generated fat data has a worse κ̂ than one trained on a clean w-tube — a
channel single-pass Φ_det omits). So Φ_det is exact single-pass and a conservative lower
bound on the recursion; the exact recursion constant needs the compounding model. None of
this touches the impossibility (v_g ≫ σ² a fortiori).

### Remark — where (R), (R′) survive, and where injection is load-bearing.

For the SINGLE-PASS floor the deterministic channel is dominant (Prop 2.5: ~70%) and the
injection channel (Prop 2.4) is the ~30% remainder — the only part needing the correlation
model. **But the ablation shows injection is load-bearing for the floor's very existence:**
with a deterministic sampler there is no floor at all (point collapse). So the clean
statement is: Theorem I(a) is the floor *of the stochastic reverse-SDE sampler* — its
existence needs the +1 injection, its being *> σ²* (rather than = σ²) needs finite κ̄, and
that latter reduction — impossibility from κ̄ < ∞ alone — remains the paper's biggest cut
in assumption surface. [Item 1: closed.]

### Theorem I(b) — the class separation, correctly attributed [REVISED 2026-07-03].

**Attribution correction (challenge #11, primary-source audit).** T1's raw jump rvar is
0.017–0.049 across t̂ and seeds — FAT, 7–20× σ², NOT ≈ σ². The 0.0031 figures in
THEORY_NOTE/SOLUTION_NOTE are post-matcher values; the matcher anchors the normal
variance by construction. **The matcher is the floor-breaker; the jump is a raw-error
reducer.** Every "jump breaks the floor" sentence in earlier notes must be read with
this correction (SOLUTION_NOTE §4c and the THEORY_NOTE Thm I caveat: amended in place
with [CORRECTION 2026-07-04] banners — done).

With the deficit channel included (earlier drafts omitted it), the jump's one-shot
output variance in the normal coordinate is

    v_jump ≈ ((1 − s(t̂)κ̂(t̂))²/a²)·V_arr(t̂) + (s²/a²) r²_res(t̂)
             + a²σ⁴/γ² + C s⁴/τ² + bias²,                               (3.1)

where V_arr is the (fat) arrival variance of the reverse flow at t̂ and κ̂ the fitted
slope there. Measured check at t̂ = 0.02: (1 − 0.141·3.83)²·0.047 + 0.0198·0.019 ≈
0.010 + 0.0004, plus bias/nonlinearity ⇒ order 0.017 ✓ (measured raw jump 0.0171).

**The true class separation is escape-by-choice versus forced traversal:**
- S_trunc must *traverse the band* [t₀, t*]: the deficit quasi-floor and the 1/s-weighted
  residual accumulation are incurred regardless of schedule (Theorem I(a)).
- S_jump has a *tunable escape*: choose t̂ at/above the band edge where BOTH (i) sκ̂(t̂)
  ≈ sκ*(t̂) ≈ 1 — the deficit becomes harmless because complete contraction only needs
  sκ̂ ≈ 1, easier at larger s (measured: 1 − sκ̂ = 0.07 at t̂ = 0.1 vs 0.46 at 0.02) —
  and (ii) r²_res(t̂) is the flat ≈ 0.03 background, entering *multiplied* by s²(t̂)
  rather than 1/s-accumulated. Larger t̂ costs curvature/mode fidelity (raw rvar 0.034
  and rbias −0.07 at t̂ = 0.2) ⇒ a measured interior optimum t̂ ≈ 0.05 — now
  mechanistically explained.
- Numbers (same net): raw jump 0.014–0.017 vs std floor 0.030 (one-shot ≈ 2×); unfixed
  recursion plateaus: jump 0.052 (T5) vs std ≈ 0.17 (LAW_b λ = 0.5) ≈ 3× — a raw
  improvement propagates to the plateau as e²/λ (Theorem II′). Post-match the samplers
  CONVERGE (jump_dscaling: jmp_mat ≈ std_mat within 9% at d = 8, equal at d ≥ 32): the
  anchor does the final work; the jump's post-match value lies in tails/bias (topped-up
  W1r 0.008–0.015 vs raw ≈ 0.05) — what a second-moment matcher cannot fix.

Honest scope: the jump's per-generation error is stated at fixed σ; its σ → 0 behavior
is folded into Corollary I(c′) below via the same capacity ratio ρ.

### Corollary I(c′) — the σ → 0 capacity law [PROVED; item 6 of the plan].

Because the floor depends on (κ̄, σ) only through ρ = 2σκ̄ (Lemma 3.2), the σ → 0 limit
is exact and clean, resolving item 6 as a *theorem*, not a scoped-out non-claim — and
as of 2026-07-04 it is UNCONDITIONAL and closed-form:

**(a) Fixed-capacity truncated sampler collapses relatively without bound —
unconditionally.** Hold the network ceiling κ̄ fixed and send σ → 0 (data → singular,
Fisher J = (d−k)/σ² → ∞). By Theorem I(a)(U) — sole assumption κ̂ ≤ κ̄ — the *absolute*
off-manifold floor tends to the closed-form constant 1/(8κ̄²) while σ² → 0, so the
*relative* collapse diverges:

    V/σ²  ≥  1/(2ρ²)·(1+o(1))  =  1/(8 κ̄² σ²)·(1+o(1))  →  ∞.

(For the measured no-overshoot class the constant is (1+3e^{−4})/8 = 0.131868 in place
of 1/8 — 5.5% apart.) The learned score's finite slope cannot track κ_max = 1/(2σ) → ∞;
the truncated sampler leaves a fixed-thickness off-manifold shell that dwarfs the
vanishing data tube.

**(b) The capacity a truncated sampler needs — explicit constant.** Bounded relative
collapse (V/σ² ≤ C) as σ → 0 *requires*, unconditionally,

    1/(2ρ²) ≤ C   ⟺   κ̄  ≥  1/(2σ√(2C)),   i.e.  κ̄ = Ω(1/σ) = Ω(√J),

now with the exact constant from Lemma 3.3(a) (and the g^{−1}(C) form for the
no-overshoot class).

This is a concrete, quotable scaling law for what Case 2 demands of the model — capacity
must scale with the singularity, which no fixed architecture provides. It is the precise,
quantitative form of "Case 2 is unwinnable for truncated samplers."

**(c) Why the fix escapes the capacity law.** The matcher (Theorem III) anchors the
normal variance to the reference's σ̂² *by construction*, independent of κ̄; it does not
need κ̄ = Ω(1/σ). The jump complements it by keeping the pre-anchor error small via the
escape t̂ ≳ 1/κ̄² (where the required slope κ(t̂) ≤ κ̄ is representable and sκ̂ ≈ 1). So
the fix converts an Ω(1/σ) capacity requirement into an O(1) anchoring operation — the
theoretical statement of why data-anchoring beats scaling the model in the singular
regime. [Consistency: the matched floor is d- and σ-robust in jump_dscaling.]

*Proof.* (a),(b) are Theorem I(a)(U) + Lemma 3.3 read at fixed κ̄ (unconditional:
g_const = 1/(2ρ²) explicitly invertible; no-overshoot: g continuous, strictly
decreasing, g(ρ) → ∞ as ρ → 0, so g^{−1}(C) exists for every C > 1). (c) is
Theorem III(a) (anchoring is a projection onto σ̂², κ̄-independent) plus the (3.1)
escape-time bound. ∎

### Theorem I(c) — the dichotomy (Case 1 vs Case 2).

The maximal slope the estimator must represent is κ_max ≈ 1/(2σ) (at s ≈ σ), and the
singular band is [0, t*] with t* the time where the score field's roughness exceeds the
pipeline's field-fitting budget. For *regular* data (σ bounded below, Fisher information
J ≍ 1/σ² bounded): a fixed adequate pipeline has ρ₀ ≈ 0 (no band), the floor (2.3)
degenerates, and annealed truncation converges — recovering Cambridge's Case 1
(their Thm 4.1). For *singular* data (σ → 0, J → ∞): any fixed pipeline eventually has
ρ₀ > 0 on a nonempty band, and Theorem I(a) applies. The transition is governed
precisely by the Fisher information through κ_max = Θ(√J): **Case 2 is not a technical
gap in their analysis; it is a different mechanism** (injection vs truncation), with a
different cure (change the sampler class or anchor the recursion — §4), not a better
schedule.

### Remark I(d) — the statistical (n-dependent) bound, non-headline.

A Fano-on-tubes two-point argument (bump M by δ on a ball of radius r: per-sample KL
≍ δ²r^k/σ², score separation ≍ δ²r^k/(σ²+s²)² localized) yields a minimax risk floor
δ²(n) ≍ σ²/(n r^k) — vanishing in n. We record it for completeness and note the measured
floors are n-flat (LAW_a, 1k→16k): at our budgets the binding constraint is the pipeline
(capacity/optimization), not information — which is why Theorem I is stated per pipeline
(Assumption R) rather than per sample size. [Challenge #4 disposition: the "weaker but
sufficient" class-restricted statement is the *right* statement, not a fallback.]

---

## §4. Theorem III — the fix [PROVED, modulo the frame-perturbation import]

### Lemma 4.1 (matcher anchoring, with shared-frame cancellation).

Let the bidirectional (per-axis) matcher use local frames estimated by kNN-PCA on a
reference corpus of size m (stale allowed), and let both the reference normal energy and
the pool normal energy be measured *in the same estimated frames*. Then the post-match
pool satisfies, per normal direction,

    v(matched pool) = σ̂²_ref + O_p(m^{−1/2}) + Δleak,

where σ̂²_ref is the reference's normal energy in those frames and Δleak is the
*difference* of tangential-leakage terms between pool and reference. First order in the
frame error θ (θ ≲ C(h/τ + σ/h + (k log m_loc / m_loc)^{1/2}), by standard local-PCA
perturbation, e.g. Davis–Kahan on the local covariance's k-th spectral gap;
Aamari–Levrard-type rates), the leakage θ²·(local tangential second moment) afflicts
pool and reference measurements *identically* and cancels in the matching equation:
Δleak = θ² · (m₂^tan(pool) − m₂^tan(ref)) = O(θ² h²) · (relative tangential mismatch),
second order. **The matcher does not need correct frames; it needs the same frames on
both sides of the equation.** [This is why the blind local-PCA fix matched the oracle fix
exactly in V1.]

**d-independence.** Every quantity in the bound (h, τ, σ, k, m, m_loc) is intrinsic to
the manifold and corpus; ambient d appears nowhere. [Postdiction, confirmed 2026-07-03:
matched floors flat in d (d^0.1: 0.170→0.210 std, 0.154→0.211 jump over d = 8→128) while
raw floors grow near-linearly (d^0.76 / d^0.95) — raw injection scales with the number of
normal directions (d−k), the matcher bound does not.]

### Theorem III(a) — memorylessness: the fix erases the recursion.

Insert the matcher after pooling (and, for the jump pipeline, after sampling). Then the
sampler's training input has normal variance σ̂²_ref + O(err) *regardless of v_{g−1}*:
the map (1.1) loses its v-argument,

    v_g = F_g(σ̂²_ref + O(err)),

a constant sequence up to O(err) — no compounding, no fixed-point equation, **no critical
fraction: stability for every λ > 0** (the matcher, not the real-data fraction, provides
the anchor; λ only sets what the sampler must learn tangentially). With S_trunc:
v_g = Φ(t₀(g)) + O(err) — truncation-only dynamics; any annealing schedule now drives
v_g to the one-shot floor: **the fix restores the hypotheses and conclusion of
2606.13796 Thm 4.1** in the singular regime. [Evidence: V6 schedule-independence; V4/
CRIT: flat 20–25 generations at λ = 0.25 and λ = 0.125 < λ*_unfixed; REC_d8/d128 flat.]

*Proof.* Lemma 4.1 makes w_g = σ̂²_ref + O(err) for all g; apply (A0)/(A2′) once per
generation; there is no g′ < g dependence left. Induction over g with the O(err) terms
summing to O(err) uniformly (they do not compound: each generation's matcher output is
re-anchored to the *same* reference). ∎

### Theorem III(b) — the JUMP+MATCH endpoint [REVISED 2026-07-03: correct attribution].

With S_jump(t̂) + bidirectional per-axis matching, the recursion's normal variance holds
at

    v_g = σ̂²_ref + O(err_matcher)   for all g,

**anchored by the matcher** (Theorem III(a) memorylessness), while the jump keeps the
quantity the matcher must correct — and the channels it cannot correct — small:

    pre-match error: (3.1) ≈ 0.014–0.017 (jump) vs 0.030–0.05 (std one-shot / growing
    with d), and post-match residual damage in non-variance channels: W1r 0.008–0.015
    (jump) — tails, radial bias, shape.

[Evidence: T1–T3 ring plateau 0.0027–0.0035 ≈ σ² for 8–25 gens incl. λ = 0.125 < λ*
(CRIT), stale-ref/provenance-free (T3) — all post-match by construction, stability and
non-variance health being the content; ZOO ellipse recursion 0.046 ≈ true 0.040; REC
d = 8..128 flat. Ablations complete the attribution: matcher OFF ⇒ 0.052 plateau (T5,
= e²_jump/λ per the law); jump OFF (std + match) ⇒ same anchored variance but larger
raw error/tails, converging to jump+match at high d (jump_dscaling).]

*Proof.* Lemma 4.1 anchors each generation's pool and output normal variance to the
reference (bidirectional: the matcher CONTRACTS when the sampler arrives fat — the ring
case, v_raw 0.017–0.048 → v_ref 0.0022 — and TOPS UP when sharpening dominates — the
MNIST thin axes; sign dictated by §2.2's three-face decomposition). Theorem III(a)
memorylessness makes the anchored state the per-generation state for all g. The jump's
(3.1) bounds the pre-match error, controlling both the matcher's correction magnitude
and the unfixed channels. Curvature: posterior-mean chord bias O(s²/τ) per point ⇒
O(s⁴/τ²) in variance [measured sphere rbias −0.07..−0.10 — bounded, not compounding;
quadratic local frames future work]. ∎

### Theorem III(c) — regime boundary and the necessity of per-axis matching.

If the normal spectrum {σ²_ax} straddles s²(t̂) (mildly singular; MNIST latents), the
sharpening factor a²σ²_ax/γ²_ax varies across axes (Prop 2.3(b)): the sampler error is
*anisotropic with mixed signs* (thin on axes σ_ax ≲ s, fat on axes σ_ax ≫ s). Any scalar
(total-energy) matcher then mis-corrects some axes in both directions; matching must be
per-axis (and bidirectional). [Evidence: T4 scalar failure thin 0.62–0.65 → MN per-axis
|thin−1| ≤ 0.012. The 0.62 value itself is a recursion+scalar-matcher compound, not the
one-step factor — challenge #9.] ∎ (immediate from Prop 2.3)

---

## §5. What is proved vs what is assumed — the honest ledger

| Ingredient | Status | Evidence anchor |
|---|---|---|
| Mixture step (Lemma 0.1) | exact, unconditional | — |
| Theorem II′ given (A0)+(A2′) | proved, elementary | LAW_b cross-prediction 1–13% |
| (A0) scalar reduction | assumed (reduced-order model) | LAW_a↔LAW_b agreement |
| (A2′) concave response | measured law + §2 mechanism | LAW_b c(rate): +0.6→0 |
| Variance ODE (Lemma 2.1) | proved (Euler limit) | exactness check vs true score |
| Orthogonality shield (Lemma 2.2) | proved, UNCONDITIONAL (definitional) | — |
| **Thm I(a) deterministic floor** | **PROVED — CLOSED FORM (σ→0), two classes: unconditional (κ̂≤κ̄ only) Φ→1/(8κ̄²); no-overshoot Φ→0.1319/κ̄²** | Lemmas 3.2/3.3; closed_form_check.py ≤2.4e-5; predicts measured 3.8σ² from measured κ̄=3.9 (finite-σ) |
| — universal law g(ρ), ρ=2σκ̄ | PROVED (Lemma 3.2 limit ODE); σ-collapse 1–8% is the convergence check | floor_theory_upgrade.py |
| — asymptotic constants | PROVED closed-form: 1/8 (uncond, exact all ρ: g=1/(2ρ²)); (1+3e^{−4})/8 (no-ovr); "0.141" = finite-σ effective value | closed_form_check.py |
| Residual-field floor (Prop 2.4) | proved in block model (R′); the ~30% *single-pass* addition to Φ_det | LAW_a shape: knee at σ², νρ₀≈3–5 |
| (R′) block/correlation model | assumed (we model) — governs only the residual channel; NOT the floor's existence (sampler-noise +1) nor its >σ² (κ̄<∞) | shape test passed |
| Thm I(b) separation | proved from above, fixed-σ scope | T1 vs LAW_a: 20–30× |
| Thm I(c) dichotomy | proved (mechanism identification) | Case 1 lit + our Case 2 data |
| Cor I(c′) σ→0 capacity law | proved (reads I(a) at fixed κ̄) | σ-collapse of g(ρ) |
| **Head-to-head vs 2606.13796** | **we observe** (5 seeds, G=20) | Arm A 9.84σ²±1.7 vs ours 1.01σ²±0.02 flat |
| **Two-horned ablation** | **we observe** (3 seeds + point-collapse verification) | noise ON 10.8–13.2σ²; OFF → single point (2D std 4e-4) |
| **Pixel-MNIST realism step** | **we observe** (3 seeds, G=8, real pixels) | unfixed 3.06 vs fixed 1.66 vs true 1.34 |
| Recursion floor constant | we model — Φ_det FP 4.54σ² is a LOWER BOUND; measured ≈11σ² (2.4× = compounding, not computed) | det_fp_diagnose + inject_ablation |
| Frame cancellation (Lemma 4.1) | proved to 1st order; PCA rates imported | V1 blind=oracle; d-flat scaling |
| Thm III(a) memorylessness | proved given Lemma 4.1 + (A0) | V4/V6/CRIT/REC flat trajectories |
| Thm III(b) jump endpoint | proved in-model | T1–T3, ZOO |
| Thm III(c) per-axis necessity | proved (immediate from Prop 2.3) | T4 vs MN |

Open mathematical items (ranked by referee risk):
1. [CLOSED 2026-07-03; REFRAMED 2026-07-04 after the ablation] (R′) is not load-bearing
   for the impossibility. Theorem I(a) is the capped-slope floor of the stochastic
   reverse-SDE sampler, assumption = "network has finite normal Lipschitz constant κ̄"
   (automatic). Precise load-bearing map (challenge #12): the sampler-noise +1 in (2.1)
   ⇒ the floor EXISTS (OFF ⇒ point collapse); κ̄ < ∞ ⇒ the floor is > σ²; (R′) governs
   only the residual channel's ~30% single-pass addition (Prop 2.4) — a strengthening,
   not a requirement. The residual referee surface there (block independence vs mixing)
   is low-stakes.
2. Lemma 4.1's PCA import: pick the exact citation form (Aamari–Levrard 2019 rates or
   Singer–Wu local PCA) and match neighborhood-size conditions to the experiments'
   (k=64, kdim as used).
3. Curvature terms carried as O(·) — one consolidated appendix computing all of them in
   tube coordinates would close the idealization gap.
4. [CLOSED 2026-07-04 — dissolved by Theorem I(a)(U).] κ̂ ≤ κ is no longer needed for
   the impossibility: the unconditional class (κ̂ ≤ κ̄ one-sided, arbitrary overshoot)
   has its own closed-form floor 1/(2ρ²)·σ² → 1/(8κ̄²), and the two classes' asymptotic
   constants differ by only 5.5%. κ̂ ≤ κ survives solely to select the tighter
   no-overshoot constant — and it is measured (rprime_probe: κ̂ tracks κ from below,
   never above).

---

## Challenge log additions (this document)

7. **THRICE-TOUCHED — final state (2026-07-03), now a THEOREM.** (a) Naive Lipschitz-cap
   argument false in the fully-relaxed idealization (V tracks s/2L → 0). (b) Restructured
   as injection-driven — but that made R′ load-bearing. (c) The resolution: the missed
   fact is that the required slope κ(t) is NON-MONOTONE (peaks at 1/2σ at t≈σ², falls to
   0 as t→0, Lemma 3.0), so the deficit band is FINITE and the frozen-in variance below
   it cannot re-thin (too few relaxation e-folds left). This makes the deficit floor a
   clean deterministic theorem (Theorem I(a), Lemmas 3.0/3.1) needing only κ̄ < ∞ — and
   it yields a universal law Φ_det/σ² = g(2σκ̄) and a σ-independent asymptotic
   Φ_det ≈ 0.141/κ̄² [both since upgraded to closed form — g explicit, exact constant
   (1+3e^{−4})/8 = 0.1319, and an unconditional variant 1/(8κ̄²); see #13 and Lemmas
   3.2–3.3]. The injection channel is demoted to a 30% correction. Lesson kept:
   the breakthrough was seeing the non-monotonicity of κ(t), which the "1/s → ∞" framing
   had hidden. deficit_floor_law.py reproduces all of it in seconds.
10. **RESOLVED (2026-07-03, poolwidth_probe.py): the small c is a measured two-channel
   near-cancellation.** Not the retention/smoothing pair I first guessed. The correct
   pair: deficit response (deterministic, dΦ_det/dw−1 ≈ −0.68, fatter pool easier to
   represent) + injection response (stochastic, ≈ +0.73, fatter pool retains more
   injected variance) = net c_total ≈ +0.05, measured concave {0.00,0.05,0.13,0.10,0.05}
   over w∈{0.0009…0.0169}, matching LAW_b. λ* ≈ 0.05 follows. See §1 Remark 2.
11. **ATTRIBUTION CORRECTED (2026-07-03, primary-source audit of T1 jsonl): "the jump
   breaks the floor" was wrong.** Raw jump rvar 0.017–0.049 (fat, 7–20× σ²); every
   ≈ 0.003 figure was post-matcher. The matcher is the floor-breaker (anchor by
   construction); the jump reduces the raw error ≈ 2× one-shot / ≈ 3× in unfixed
   recursion and keeps tails/bias small (post-match W1r 0.008–0.015 vs raw 0.05).
   Theorem I(b)/III(b) restated accordingly (escape-by-choice vs forced traversal;
   matcher-anchored endpoint). SOLUTION_NOTE §4c and THEORY_NOTE's "jump sits 6× below
   Φ(0+)" caveat MUST be amended before any paper draft uses them.
8. LARGELY RESOLVED (2026-07-03): R′ formalized as the block-correlation model; Prop 2.4
   proved within it, and the resulting floor shape νρ₀(σ² + t₀) passed the LAW_a
   confrontation (knee at t₀ ≈ σ², saturation, n-flatness) with the constant effective
   to ±2× (νρ₀ ≈ 3–5; drift direction consistent with r²(t) rising toward the band
   edge). Residual openness: block independence could be relaxed to mixing; negative
   association is the only direction that could break the bound — discuss in the paper.
9. RESOLVED NEGATIVE (2026-07-03, checked before claiming): the parameter-free one-step
   sharpening postdiction FAILS quantitatively. MNIST thin-axis variances are 0.151–0.206
   (mnist_latents32.npz), so a²σ²_ax/(a²σ²_ax + s²(0.02)) ≈ 0.88–0.91 per generation —
   not the measured 0.62–0.65. The measured value is a *recursion* fixed point
   (compounded sharpening ≈ 0.90/gen mixed with λ real gives ~0.81 at λ=0.5) further
   squeezed by the scalar matcher's global contraction (fat-axis-driven β < 1 applied to
   all axes) — consistent in sign and mechanism, but with a fitted component, NOT
   parameter-free. Paper claim downgraded: the sharpening formula predicts the
   *direction and the regime boundary* (σ_ax ~ s(t̂) ⇒ anisotropic thinning ⇒ per-axis
   matching required — confirmed by MN arm), not the numerical thin_ratio. Prop 2.3
   stays (it is exact in the model); its role is mechanism, not postdiction.
12. **BREAK-TEST, PREDICTION FALSIFIED then REPLACED (2026-07-04, inject_ablation.py +
   verify_nonoise.py).** I predicted the no-injection recursion would land on the
   deterministic floor Φ_det = 4.54σ². **Wrong.** It collapses to a *single point*
   (2D std → 4·10⁻⁴, NaN-free), i.e. V → 0, *below* σ². Root cause of my error: I had
   called Φ_det "deficit-only, injection omitted," but Φ_det integrates the variance ODE
   (2.1) *with* its +1 injection term — it is the with-injection floor. Correct mapping:
   injection-ON ↔ Φ_det (measured 10.8–13.2σ² at fixed t₀; Φ_det recursion FP 4.54σ² is a
   2.4× *lower* bound, gap = per-gen compounding); injection-OFF ↔ the same ODE minus the
   +1 ⇒ V→0 ⇒ point collapse. Net effect: a STRONGER result — self-consuming collapse is a
   two-horned trap (deterministic → point; stochastic → fat floor; neither reaches σ²; the
   matcher fixes both). The +1 injection is load-bearing for the floor's *existence*, finite
   κ̄ for its being *> σ²*. §3 empirical subsection + the Remark updated to this framing.
   Lesson: name the exact term in your own equation before claiming an ablation will isolate
   it — I mislabeled the +1 and the data caught it.
13. **STATEMENT/NUMBERS MISMATCH CAUGHT, RESOLVED BY CLOSED FORMS (2026-07-04,
   floor_theory_upgrade.py + closed_form_check.py).** The 2026-07-03 Theorem I(a) proof
   bounded via the *constant*-κ̄ comparison (V ≥ V_{κ̂≡κ̄}) while its verified table was
   computed with κ̂ = min(κ*, κ̄) — two different comparison classes (a net with κ̂ ≤ κ̄
   may still *overshoot* κ* where κ* is small, thinning further than the min-profile).
   Resolution, and a large net strengthening: (i) rescaling t = σ²u, V = σ²W collapses
   the ODE to a σ-free limit ODE in ρ = 2σκ̄ alone (Lemma 3.2) — universality becomes a
   theorem; (ii) BOTH class envelopes integrate in closed form (Lemma 3.3): unconditional
   g_const(ρ) = 1/(2ρ²) EXACTLY (Φ → 1/(8κ̄²); floor > σ² ⟺ ρ < 1/√2), no-overshoot
   g(ρ) explicit with C₀ = (1+3e^{−4})/2, Φ → 0.131868/κ̄². The previously quoted
   asymptotic constant 0.141 was the finite-σ effective value of the exact 0.131868
   (+5–8% = the measured O(σ²/ρ²) correction — consistent in direction and size). The
   two classes' constants differ by 5.5% ⇒ the overshoot question (open item 4) is
   dissolved, and Cor I(c′)'s σ→0 impossibility is now UNCONDITIONAL (κ̂ ≤ κ̄ one-sided
   is the sole assumption) with explicit constants. My own hand-predicted inner-ODE
   boundary value Ω(4) = 2 for const mode was also wrong (slaved solution is √y + ½, not
   √y) — caught because the two independent computations disagreed (0.4908 vs 0.5000);
   the exact integrating-factor solution superseded both. Verified: closed forms match
   direct limit-ODE integration to ≤ 2.4×10⁻⁵ across ρ ∈ [0.1, 1].
