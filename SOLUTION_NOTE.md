# Estimation-Driven Collapse on Manifolds: Law, Critical Fraction, and a Data-Only Fix

Working note, 2026-07-02. All numbers reproducible from `solution_law_test.py` /
`solution_validation.py` (checkpoints in the matching `.jsonl` files). Ring: R=2.5, σ=0.05,
4000 samples, 50% real unless stated; nets 256-wide MLP score, VP-OU diffusion, Euler reverse,
K=200 steps; anneal2 schedule t0(g)=0.5/(1+g)² unless stated.

## 1. The gap (Khelifa–Turner–Venkataramanan 2606.13796, Appendix E Case 2)

Their theory: recursive collapse is driven by reverse-process truncation smoothing; annealed
truncation t0→0 provably eliminates it — but only under finite Fisher information J(p_data)<∞.
For singular / manifold-supported data (J=∞) they state annealing "is not justified" and leave
the regime open. Verified empirically in earlier sessions: with their fix on, manifold data
plateaus at offman≈0.29 vs fresh floor ≈0.19, flat across 4× σ and 3.5× capacity.

## 2. The mechanism and the law

On singular support the per-generation sampling excess in the normal direction,
e0² (score-estimation at small t + solver discretization), cannot vanish: floor is flat in
dataset size (1k→16k) and capacity, drops only partially with solver steps, leaving an
irreducible core (E2). The self-consuming loop recycles e0² — the driver *switches* from
truncation (annealable) to estimation (not annealable). Annealing addresses the wrong term.

Normal-variance recursion (linearized): pool w_g = λσ² + (1−λ)v_{g−1};
sampled v_g = w_g + e²(w_g) + s²(t0(g)), with e²(w) = e0² + c·w. Then

    v_g = ρ v_{g−1} + β,   ρ = (1+c)(1−λ),
    v_∞ = ((1+c)λσ² + e0²) / (1 − ρ) + s²_res,   critical fraction λ* = c/(1+c).

Below λ* no plateau exists: variance is self-sustaining (observed: λ=0.125 trajectory pinned
at ~0.56 while annealing input collapsed 0.056→0.008; λ=0.25 unfixed still fat, offman 0.40,
at g=19). Fixed points measured: λ=0.75→v 0.090, 0.5→0.154, 0.25→0.36↓(slow), 0.125→stall;
e0² fitted 0.044–0.057, independently consistent with the measured fresh floor 0.057.

## 3. The fix: moment-matched normal contraction (data-only)

Each generation, before training: estimate local frames by k-NN PCA against the REAL fraction
(no manifold knowledge); contract synthetic samples' signed normal offsets by
√(v_real_n / v_syn_n) so the synthetic normal variance matches the real fraction's. Requires
only the training pool + provenance labels. No external corrector, no oracle geometry.

**Lemma 1 (severed channel), observed directly** (V1 pool log, per generation
v_real / v_syn_before / v_syn_after): synthetic arrives up to 0.573 (25–270× fatter than real
0.002) and is matched to 0.003–0.007 every round → v_g no longer depends on v_{g−1}:
v_g = σ² + e0² + s²(t0(g)). Recursion eliminated for ALL λ>0; no critical fraction under the fix.

**Consequence:** the fixed system follows truncation-only dynamics — the regime Cambridge's
Thm 4.1 provably handles — so annealing removes the remaining s² term. Moment matching and
annealed truncation compose: matching kills the estimation channel, annealing the truncation
channel. Residual = the one-shot sampling floor, which is not a recursion phenomenon (one-shot
manifold literature, e.g. 2505.09922, applies).

## 4. Validation battery (all pre-registered, all passed)

| Objection | Test | Result |
|---|---|---|
| Oracle geometry used | V1 blind local-PCA fix, ring | 0.1917 vs oracle 0.1904 vs floor 0.1909 — identical |
| Circle symmetry trick | V2 ellipse (a=3,b=1.5, varying curvature) | fresh 0.207 / unfixed 0.313 / fix 0.213 — 95% of increment erased |
| Only k=1 | V3 sphere S² in R³ | fresh 0.202 / unfixed 0.379 / fix 0.226 — 86% erased (residual = k-NN frame quality on 2-D patches; rvar 0.082 vs 0.069) |
| Slow leak | V4 λ=0.25, G=20 | fix flat 0.208→0.188 across 20 gens (tracks s² exactly: rvar 0.060 ≈ 0.059+s²); unfixed 0.40 and biased −0.17 |
| Tangential damage | V5 angular KS | fix 0.035–0.037 vs unfixed 0.039 — fix adds nothing; mild angular clumping is a shared, non-singular channel (existing literature's regime) |
| Schedule-specific | V6 anneal1 | fixed trajectory tracks s²(g)+0.059 within 4–8% every generation |
| Bias channel | rbias | −0.05..−0.30 unfixed → +0.001..−0.03 fixed |

## 4b. Scale experiment (2026-07-02 morning, scale_experiment.py + scale_variants.py)

| Setting | fresh floor | unfixed | best blind fix | increment erased |
|---|---|---|---|---|
| S² curved-embedded, d=16 | 0.827 | 1.295 | 0.867 (local frames) | 91% |
| S² curved-embedded, d=64 | 2.361 | 3.031 | 2.388 (local frames) | 96% |
| MNIST 32-d PCA latents | thin 1.108 | 1.539 | 1.122 (local, kdim=13) | 97% |

- Floor grows superlinearly with ambient d (0.19 → 0.83 → 2.33 for d=2→16→64; metric
  sanity-checked: true data scores 0.058) — estimation dominates *harder* at scale, so
  annealing is even more mis-aimed in realistic dimensions. Lemma-1 logs at d=64: synthetic
  pool arrives 313x fatter than real (5.76 vs 0.018), matched down every generation.
- MNIST instantiation lesson: global-axis matching (thin-8: 59%, all-32: 61%) fails on a
  curved real manifold; local kNN-PCA frames recover 97%. It's local-vs-global, not
  axis-count. knn_excess: local fix recovers 66% of that increment — remainder is
  tangential/mode drift, the non-singular channel out of scope by design.

## 4c. JUMP+MATCH — the upgraded solution (2026-07-02, jump_solution_test.py/.jsonl)

> **[ATTRIBUTION CORRECTION 2026-07-04 — challenge #11, primary-source audit. Read before
> quoting anything below.]** Every ≈0.003 rvar in this section is a **post-matcher** value;
> the RAW jump output is 0.017–0.049 (fat, 7–20× σ², per the T1 jsonl itself). So "the
> floor is broken" sentences must be read as claims about the JUMP+MATCH *pipeline*, in
> which **the matcher is the floor-breaker** and the jump is a raw-error reducer (≈2×
> one-shot, ≈3× unfixed plateau — T5's 0.052 vs std ≈0.17). Post-match, std+match and
> jump+match converge to the same anchored variance (head_to_head 2026-07-04: both 1.01σ²,
> 5 seeds); the jump's post-match value lies in tails/bias (W1r), not variance. Corrected
> statements: PROOFS.md Theorems I(b) and III(b).

Sampler: reverse SDE only to moderate t_hat, then Tweedie jump x0=(x−s·eps)/a — never
consults the small-t score, evading the singular-support estimation floor (one-shot
optimality of full denoising for singular data: arXiv 2503.12966 — cite as building block).
Matcher: BIDIRECTIONAL local moment matching (contract fat by scaling / top-up thin by
additive normal-complement noise) against a 2000-sample reference corpus; provenance-free
variant matches the whole unlabeled pool against ONE stale (g=0) reference.

Results (ring, sigma^2=0.0025; old K200 fresh floor rvar=0.0642, old best fix plateau 0.065):
- T1 anatomy: topped-up jump rvar 0.0030–0.0033 at every t_hat in {0.2,0.1,0.05,0.02};
  best W1r at t_hat=0.02. THE FLOOR IS BROKEN [by the pipeline; matcher does the anchoring
  — raw jump in this same jsonl is 0.017–0.049, see banner]: 20x below K200 floor, 11x below K800.
- T2 recursion (labeled, lam=0.5): plateau rvar 0.0033–0.0035, W1r 0.011–0.013 —
  self-consuming training now ends at ~true distribution statistics (1.3x sigma^2),
  19x BELOW the old fresh floor. Recursion with JUMP+MATCH beats non-recursive standard
  sampling.
- T3 provenance-free (stale reference): 0.0033 at lam=0.5 (indistinguishable from labeled,
  <3% diff); holds at lam=0.25 (0.0032, W1r 0.018). No provenance labels needed.
- T5 ablation (no matcher): plateau rvar 0.0518 — fat, NOT thin: raw-jump estimation excess
  (e_jump^2 ~ 0.02) dominates posterior sharpening at ring scales, and the law predicts the
  plateau perfectly: e_jump^2/lam ~ 0.049 vs observed 0.052. (My pre-registered death-spiral
  direction was wrong at these parameters — recorded as falsified; bidirectionality remains
  necessary, see MNIST.) Ablation independently CONFIRMS the law with the new sampler's e0.
- T4 MNIST: FAILED two-sided criterion — thin_ratio 0.62–0.65 (under-dispersed), and
  matching made the fresh jump WORSE (0.89 raw -> 0.71 matched). Diagnosis from the data:
  jump error on MNIST is ANISOTROPIC (thin-axis sharpening in the sigma_normal ~ s(t_hat)
  regime, fat elsewhere from estimation) and the scalar total-energy matcher contracts when
  it should selectively inflate. Regime boundary identified: JUMP+MATCH wins where
  sigma_normal << s(t_hat) (strongly singular — exactly the Case-2 regime); standard
  sampler + local matching (4b) remains best in the mildly-singular regime (MNIST latents,
  thin 1.12). Fix path if wanted: per-direction (anisotropic) local matching.

Verdict: in the strongly singular regime the upgraded PIPELINE does not just remove the
recursive increment — it removes the floor [attribution: the MATCHER anchors, the jump
reduces raw error — see banner above], provenance-free, at 75% synthetic, ending at
true-distribution statistics. In the mildly singular regime use 4b.

## 5. Honest boundaries

- "Proof" here = empirically verified premises + a conditional argument. The formal theorems
  (minimax lower bound e0(t0, n) ≳ f on singular support ⇒ impossibility for any truncation
  schedule; the law with rates; fix ⇒ Thm 4.1 hypotheses restored) are the paper's math, to be
  written and proved by Naman.
- Sphere residual 14% of increment: corrector estimation quality (k=50 naive PCA frames),
  not a fundamental leak — larger k / quadratic local fits expected to close it.
- v_syn_after (0.003–0.007) sits slightly above v_real (0.002): finite-k + normal-direction
  estimation leakage; small vs floor 0.057. Self-inclusion in real-vs-real kNN biases v_real
  ~4% low (argpartition; fix in paper version).
- Rate-fit slope −0.5 anomaly in the law test: polynomial annealing transient contaminates
  the geometric-rate estimator; fixed-t0 rate experiment would clean it.
- Provenance labels assumed (you know which training samples you generated) — same assumption
  as the real-data-anchor literature (Bertrand et al.).
- Tested at d≤3, k≤2, σ=0.05, N=4000, one architecture family. High-dim image-scale validation
  (latent-space of a real diffusion model) is the natural next experiment.

## 6. Prior-art position (searched 2026-07-02, all gates passed)

- 2606.13796 (Cambridge): truncation-only mechanism, Case 2 explicitly open, no follow-ups yet.
- 2602.16601: generic discounted error accumulation with mixing — cite; requires absolutely
  continuous data (structurally excludes this regime), t0 fixed, bounds not laws; its own
  Eq. 12 (ε² ~ n⁻¹ t0^{−d/2}) contains the seed of the impossibility argument, never connected.
- 2605.20235: one-shot manifold score theory — building block for the lower bound.
- Fix lane unclaimed (nearest: NeurIPS-2024 synthetic-data debiasing for inference;
  2402.07087 self-correction needs an external corrector; SIMS negative guidance; rectified-flow
  mitigation 2412.08175).

## 7. Claims for the paper

1. Impossibility: on singular support, no truncation schedule eliminates recursive collapse
   (estimation term non-annealable).
2. Law: v_∞ = ((1+c)λσ² + e0²)/(λ+cλ−c) with critical real-data fraction λ* = c/(1+c) —
   the first quantitative characterization of the Case-2 limit.
3. Fix: real-anchored moment-matched normal contraction severs the compounding channel for all
   λ>0, restores Thm 4.1 dynamics, and returns the system to the one-shot floor — blind to
   geometry, data-only, composes with annealing.
