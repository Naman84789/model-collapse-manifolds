# Plain-English walkthrough of the paper

*A study companion. The goal is that after reading this you can walk through every
theorem, explain every assumption in plain words, and answer "why?" for each proof
step — in front of a professor, a reviewer, or an audience — without the paper open.*

Read this alongside `main.tex`. Each section here maps to a labelled result there.
The rule of the paper is the three-tier honesty legend, and it applies here too:
**we prove** (math, given stated assumptions), **we observe** (multi-seed experiments),
**we model** (measured laws we do not derive). If someone asks "is this proved or
measured?", the answer is always one of those three, and you should know which.

---

## 0. The whole paper in six sentences

1. Train a generator, sample from it, train the next generator on those samples, repeat —
   quality drifts, like a photocopy of a photocopy. That is **model collapse**.
2. The best prior theory (Cambridge, Khelifa–Turner–Venkataramanan 2606.13796) proves you
   can repair the drift by annealing the sampler's early-stopping time — **but only when the
   data is smooth** (finite Fisher information). They explicitly leave open the case that
   describes real images/audio: data on a **thin manifold**.
3. **Negative result:** for manifold data there is a hard **floor** on how faithful a
   fixed-size model can stay, set by one number (the steepest slope the network can
   represent), and **no schedule removes it**. In the thin limit the floor is closed-form,
   `1/(8·κ̄²)`.
4. **Constructive result:** a cheap **anchor** — each round, rescale the model's output so
   its fine-scale width matches a small *fixed* sample of real data — makes the loop
   **memoryless**, so error stops compounding for *any* positive fraction of real data.
5. **Evidence:** 20-generation head-to-head, the prior method plateaus at 9.8σ², the anchor
   holds 1.01σ²; on MNIST pixels the anchor removes ~81% of the collapse; a CIFAR-10 pixel
   run (3,072-dim, 3 seeds) reproduces the separation with full mode coverage.
6. Everything runs on CPU and is reproducible from released scripts.

**If you only remember one thing:** the floor lives in a term the schedule can't touch, and
the anchor kills that term by changing *what the model trains on*, not *how much real data*
it sees.

---

## 1. The setup — the objects you must be able to define cold

**The data model (σ-tube).** Real data sits on a `k`-dimensional surface `M` in `d`-dim
space, blurred by a Gaussian of width `σ` in the `d−k` directions perpendicular
("normal") to the surface. `σ → 0` is the singular/manifold limit. As `σ → 0` the Fisher
information `J ≈ (d−k)/σ²` blows up — that divergence is *exactly* what makes the estimator's
job impossible, and it's Cambridge's "Case 2."

**Why we can work in one dimension.** Collapse is about *width perpendicular to the
surface*. Fix a point on `M`, look along one normal direction. There the data is just a 1-D
Gaussian `N(0, σ²)`, and the ideal denoiser is a straight line `ε*(x,t) = κ(t)·x`. So the
whole problem reduces to tracking a single number, the normal variance, through the
diffusion. Everything technical is about that one number.

**The one function that runs the whole paper: κ(t).** The slope the network must represent,
`κ(t) = s(t)/γ²(t)`. Its shape is the engine of the negative result:
- starts at 0 (at `t=0` there is no noise to remove),
- rises,
- **peaks at `κ_max = 1/(2σ)`** near `t ≈ σ²`,
- falls **back to 0** as `t → 0`.

That non-monotone bump is the crux. Say it out loud: *the required slope is huge but only in
a brief window of noise levels.* Draw the bump on a napkin. If you can't, you can't explain
the paper.

**The recursion.** Pool for generation `g` is `λ`·(fresh real) + `(1−λ)`·(previous
generation). Because variance is linear in the mixture, the pool width is exactly
`w_g = λσ² + (1−λ)v_{g−1}`. The output width is `v_g`. This bookkeeping is exact, no
assumption.

---

## 2. The two assumptions — know these better than the theorems

The theorems are *proved*. But they rest on two things we **measure**, not derive. A
reviewer will push hardest here, so this is where your understanding must be strongest.

### Assumption A0 (scalar reduction): `v_g = F_g(w_g)`
**Plain English:** the output width depends on the input pool only through the pool's
*width*, one number in, one number out.

**Why it's reasonable (not circular):** the collapse observable is a variance along the
normal fibre — it is *one number per axis by construction*. This is an **order-parameter**
statement, the same move as mean-field physics: the magnetization closes on itself even
though the microscopic spins are high-dimensional. We are **not** claiming the network is
one-dimensional; we're claiming the *width* closes on itself.

**How it's validated:** fit the law in one experiment (one `t₀`), use it to predict the
fixed point of a *different* experiment (another `t₀`). Predictions land within 1–13% across
six cells. If A0 were false, cross-prediction would fail.

**What would break it:** if the output width depended on the *shape* of the pool, not just
its width. We test the sharpest version of this in the gap-closure work (the mixture test,
§8) and it holds to within measurement error.

### Assumption A2 (concave response): `F_g(w) = a²w + s² + e²(w)`, with `e²` concave, `e²(0) > 0`
**Plain English:** output width = (pool width shrunk by the forward map) + (truncation
leftover) + (a sampler-added excess `e²`). The excess has three properties, and you should be
able to justify each:
- **`e²(0) > 0`** — even a perfectly thin pool comes out too fat. *This is the floor.* It's
  positive because the network can't represent the peak slope no matter how thin the input.
- **increasing** — a fatter pool gives a fatter output. Obvious.
- **concave (saturating)** — doubling the pool width less than doubles the added excess.

**Why concave (the "why" a professor will ask):** the excess is the network failing to
represent the steep part of `κ`. A *fatter* pool has a *smaller* peak slope
(`κ_max = 1/(2σ_pool)` falls as `σ_pool` grows), so it's *easier* to represent, so each
extra unit of pool width buys *less* extra error. Diminishing returns = concave. This isn't a
convenience assumption; it's forced by the same `κ`-bump geometry as the floor.

**How it's validated (two independent estimators — get the direction right, a reviewer
will check):** concave means the slope of `e²` is *larger at thin pools and smaller at
fat pools* (diminishing returns). That is what's measured, both ways. (1) The direct
probe (train single-width pools, measure `dv/dw − 1`) gives a small slope that falls
`+0.13 → +0.05` across the upper half of the measured range — where all the recursion's
fixed points live. (2) The recursion's own convergence rates give an independent
estimate falling from `+0.6` at the *thinnest* fixed-point pool (`λ=0.75`) to `≈0` at
the *fattest* (`λ=0.25`) — same saturating direction. The two estimators agree in sign
and trend but not magnitude where they overlap; the paper treats the rate-based one as
directional only (rate fits must separate a transient — a known failure mode). At the
probe's very thinnest widths the slope estimates rise from 0; that regime is never
visited by the recursion, and the law only needs a unique downcrossing of `T(v) − v`,
which the measured slopes (≤0.13 at every fixed point, vs. thresholds `λ/(1−λ) ≥ 1/3`)
guarantee. And the slope's small net value is a near-cancellation of two ~0.7-sized
channels — see §8.

---

## 3. Theorem 1 (the law of collapse) — the dynamics

**Statement in words.** Under A0+A2 the recursion is one-dimensional: `v_g = T(v_{g−1}) + s²_g`.
- **(i)** If you keep enough fresh data (`λ > λ* = c∞/(1+c∞)`), there is a **single stable
  plateau** `v∞`, and the width converges to it from *any* start and *any* schedule.
- **(ii)** Below that threshold (`(1−λ)(1+c∞) > 1`), **no plateau** — width runs to infinity.
- **(iii)** If `e²` is a straight line, `v∞` has a clean closed form.

**The one thing to notice:** `λ*` and `v∞` depend on the sampler excess `e²` *only*. The
truncation schedule enters *only* as the additive `s²_g`, which `→ 0`. So:

> **Remark (why annealing can't help singular data).** Cambridge's remedy anneals `s²_g` to
> zero. But the plateau is set by `e²`, which no schedule touches. Their Case 1 is the lucky
> case `e² ≡ 0` (smooth data has no floor), so annealing lands them on `σ²`. Case 2 is
> exactly the regime where `e²` has a floor. **This remark is the paper's thesis in one
> paragraph — be able to say it cold.**

### How the proof works (Appendix A), step by step, and *why* each step
Define `g(v) = T(v) − v`. You want to show it has one root and iterates fall onto it.
1. **`g` is concave, `g(0) > 0`.** *Why:* `T` is affine composed with `w + e²(w)` (concave,
   since `e²` is), minus a linear term. `g(0) = λσ² + e²(λσ²) > 0` because `e² > 0`. → the
   curve starts above zero and bends down.
2. **`g` eventually has negative slope when `λ > λ*`.** *Why:* `g'(v) → (1−λ)(1+c∞) − 1 < 0`.
   A concave curve that starts positive and ends decreasing must cross zero **exactly once**
   (concavity forbids a second crossing). → existence + uniqueness of `v∞`.
3. **Contraction above `v∞`.** *Why:* for `v ≥ v∞`, `T'(v∞⁺) < 1`, so `T` pulls you toward
   `v∞` — a standard contraction, and the perturbations `s²_g → 0` can't stop it (split the
   error sum at `j = g/2`; the standard perturbed-contraction trick).
4. **Barrier below `v∞`.** *Why:* if an iterate dips below `v∞`, the *unperturbed* map
   `u_g = T(u_{g−1})` climbs monotonically back up to `v∞`, and the real (perturbed) iterate
   sits above it because `s²_g ≥ 0`. → `liminf ≥ v∞`. Combined with the top barrier,
   `v_g → v∞`.
5. **(ii) Divergence.** *Why:* if `(1−λ)(1+c∞) > 1`, then `g' > 0` everywhere, so
   `g(v) ≥ g(0) > 0` always — every step adds at least `T(0)`, and the width marches to
   infinity regardless of schedule.
6. **(iii)** Plug the affine `e² = e₀² + cw`, solve the linear fixed point. Algebra.

**If a professor asks "why is there a critical fraction at all?"** Because each generation
the synthetic majority `(1−λ)` re-injects the old width. If real data `λ` isn't large enough
to overwrite that inheritance, the width feeds on itself and diverges. `λ*` is the break-even.

---

## 4. Theorem 2 (the closed-form capacity floor) — the impossibility

This is the headline "we prove." It's built from four lemmas; know what each *does*.

**The chain of lemmas (say what each buys you):**
- **Lemma (variance ODE).** One reverse Euler step is affine-in-`x` + independent noise, so
  the variance obeys `−dV/dt = (1 − 2κ̂/s)V + 1`. *Why it matters:* it reduces the whole
  sampler to one scalar ODE. The `+1` is the sampler's own injected noise — remember that,
  it's load-bearing later. With the *exact* slope, `V = γ²` solves it exactly (sanity check).
- **Lemma (orthogonality shield).** A real score net is wildly nonlinear. Split it into its
  best affine fit + orthogonal residual `δ`. The normal equations force
  `E[δ] = E[xδ] = 0`, so `δ` has **zero first-order effect** on the variance and can only
  *add* variance. *Why it matters:* this is the answer to "your floor is just an artifact of
  pretending the score is linear." No — the nonlinearity lives entirely in `δ`, which can
  only make the floor *bigger*. So the affine lower bound is a genuine lower bound for the
  full nonlinear network. **This is the most important lemma to internalize** because it's
  the most natural attack. One honest scope note (a sharp reviewer may push here): the
  normal equations hold under the *true* marginal `p_t`; along the sampler's own path the
  ensemble drifts away from `p_t`, and the cross term re-enters as (ensemble deviation) ×
  (residual size `r`). It stays second order because `r` is *measured* small
  (`r² ≈ 0.02–0.05`) — so the shield is "we prove, first order," not "unconditional."
- **Lemma (finite band).** Because `κ(t)` peaks and comes back down, any ceiling `κ̄ < κ_max`
  is exceeded only on a **finite band** of times. *Why it matters:* the network only fails in
  a short window; but variance frozen in that window can't be removed later, because below
  the band only `≤ ρ²/2` contraction e-folds remain. Too little runway to recover.
- **Lemma (singular-limit ODE).** Rescale `t = σ²u`, `V = σ²W`. The ODE becomes `σ`-free:
  `−dW/du = 1 − (2m(u)/√u)W`. *Why it matters:* **universality is now a theorem, not a
  coincidence** — the floor in units of `σ²` depends only on `ρ = 2σκ̄`, nothing else.

**The theorem itself.** With slope ceiling `κ̄` and `ρ = 2σκ̄`:
- **(U) Unconditional** (no assumption on the net): `V ≥ σ²/(2ρ²) = 1/(8κ̄²)`. Exceeds `σ²`
  whenever `ρ < 1/√2`.
- **(N) No-overshoot** (net never over-contracts, which we measure): tighter constant
  `(1+3e⁻⁴)/8`. The two constants differ **5.5%** — the distinction is minor.

### Why the closed form is what it is (Appendix C), the steps that matter
- **(U) derivation.** Comparison principle: smaller slope ⇒ larger variance, so worst case is
  `κ̂ ≡ κ̄`. Plug the constant envelope into the limit ODE; integrating factor `e^{−2ρ√u}`
  gives `W(0⁺) = ∫₀^∞ e^{−2ρ√u} du = 1/(2ρ²)`, **exactly**, for every `ρ`. *Why the
  comparison principle is legal:* same forcing, same data, monotone dependence on `κ̂` — a
  standard scalar ODE comparison.
- **(N) derivation.** The band edges solve `√u/(1+u) = ρ/2`, a quadratic. Above the band the
  solution is the exact "slave" `W = 1+u`. Integrate across the band with the same
  integrating factor to get `W_exit`; below the band the equation is uncapped with solution
  `(1+u) + A(1+u)²`; match at the lower edge, read off `A`, and `g(ρ) = 1 + A`. Every step is
  "solve a linear ODE on a sub-interval and match at the boundary."
- **The `σ → 0` limits.** `ρ → 1⁻`: band shrinks to a point, `g → 1` (no floor when the net
  can almost reach the peak). `ρ → 0`: `g ≈ C₀/ρ²`, the floor blows up — the thinner the
  manifold, the relatively worse.

**Corollary (the actual answer to Case 2).** Hold `κ̄` fixed, `σ → 0`: `V/σ² → ∞`. A
fixed-size model collapses *without bound*. To stay bounded you need `κ̄ = Ω(1/σ) = Ω(√J)` —
capacity must grow with the singularity. **No fixed architecture provides it.** That sentence
is the paper's answer to the open problem; be able to say it in one breath.

**The confrontation (we observe) — corrected 2026-07-10.** Measure one real network's
slope ceiling: `κ̄ ≈ 3.7–4.3`. The floor law's envelope value at that ceiling is
`3.8–4.6σ²` — a **proven lower bound**. The directly measured single-pass floor is
`6.3–7.5σ²`: the bound holds and its envelope explains ~60% of the measurement; the
rest is two *measured* channels (in-band profile sag `+0.7σ²`; ensemble displacement
`+2.6σ²` — the sampler's own ensemble is fatter than `p_t`, so it probes the S-shaped
response into its flat tails and experiences a smaller slope). The most persuasive
evidence is now **interventional**: five training protocols move `κ̄` from 3.5 to 7.5
and the measured floor falls 7.8 → 2.5σ², tracking `1/κ̄²` and never crossing the
bound. (An earlier version of this walkthrough claimed the measured floor itself was
`3.8σ²` — that number was the envelope prediction, not the direct measurement.)

---

## 5. Theorem 3 (anchoring makes the recursion memoryless) — the fix

**Statement in words.** Collapse happens because `F_g` has `v_{g−1}` inside it — each
generation inherits the last one's width, so errors compound. The matcher **overwrites the
pool's normal width with a fixed number (the reference's) before training**, so the input to
`F_g` no longer depends on `v_{g−1}`. The recursion becomes a constant sequence. Consequences:
**no fixed-point equation, no critical fraction, stable for every `λ > 0`.** With the standard
sampler it *restores Cambridge's Case-1 dynamics* in the singular regime — i.e. it turns the
unsolvable case back into their solvable one.

**The operator, concretely.** For each point: find its `k` nearest neighbors in a small
*stale* reference (2,000 real samples drawn *once* at generation 0); build the local PCA
frame (top directions tangent, rest normal); rescale the point's normal coordinates so the
pool's local normal second moments equal the *reference's* in the *same frames*. It is
**bidirectional** (contracts fat axes, inflates thin ones) and **per-axis**.

**Why bidirectional and per-axis (not a knob, forced by theory):** the sampler's error is
*signed and anisotropic* — fat from noise injection, thin from posterior-mean sharpening
(Prop, Appendix). A scalar or one-sided matcher fixes some axes and worsens others. Measured:
scalar matcher leaves MNIST thin-ratios at 0.62; per-axis gets `|thin−1| ≤ 0.012`.

### The lemma the fix rests on (shared-frame cancellation)
**What it says:** even though the PCA frames are estimated with error, the *same* error hits
the pool and the reference (frames come from the reference), so at first order it **cancels**.
The leftover error is `O(m^{−1/2})` (finite reference sample) plus a *second-order* frame
term. Crucially **the ambient dimension `d` appears nowhere** — the bound is intrinsic
(`k, τ, σ, m`).

**Why `d`-independence matters, and the postdiction that confirms it:** the floor is set by
*local normal slope*, which doesn't know the ambient dimension. So the fix should be flat in
`d`. Test: across `d = 8 → 128`, raw floors grow near-linearly (`~d^0.76`) — that's the
collapse — while *matched* floors are flat (`~d^0.1`). We predicted flat before measuring;
it's flat.

### Why this is NOT a replay buffer (the #1 reviewer question — memorize this)
Replay/accumulation remedies **re-insert real samples** — they preserve *samples* and change
`λ`. The anchor preserves the *local normal moments* of a fixed stale reference and imposes
them on **whatever the model currently generates** — it changes `F_g`. The distinction has
teeth: a replay buffer still lets the `(1−λ)` synthetic majority set the pool's fine-scale
statistics, so the floor still governs that majority. Anchoring **overwrites the majority's
fine-scale statistics**, which is why it *removes* the floor instead of *diluting* it. In the
language of the law: **replay changes `λ`; anchoring changes the response map `F_g`.**

**Attribution, stated plainly (don't oversell the jump):** the *matcher* is the floor-breaker;
the Tweedie jump is only a raw-error reducer (its own raw output is fat, 7–20× σ²). Post-anchor
the standard and jump samplers land on the same variance. We report both.

---

## 6. What is proved vs. measured vs. modelled — the honesty ledger

Be able to place every claim:

| Claim | Tier | One-line justification |
|---|---|---|
| The floor exists and is `1/(8κ̄²)` in the limit | **prove** | comparison principle + exact ODE integral |
| Universality in `ρ` | **prove** | the rescaled ODE is `σ`-free |
| `σ→0` needs `κ̄ = Ω(√J)` | **prove** | invert the floor bound |
| Anchoring kills the memory / critical fraction | **prove** (given A0 + frames lemma) | the `v_{g−1}` argument drops out |
| `d`-independence of the fix | **prove** (frames) + **observe** (d-sweep) | intrinsic bound; flat measured floors |
| Two-horned trap (noise on = fat, off = point) | **observe** | ablation, 3 seeds |
| Head-to-head 9.8σ² vs 1.01σ² | **observe** | 5 seeds, 20 gens |
| CIFAR coverage 1.00× (no mode collapse) | **observe** | 3 NN metrics, 3 seeds |
| A0, A2 hold for the real pipeline | **model** | cross-validated, not derived |
| The 2.4× recursion residual | **model** | partly closed empirically (§8), rest open |

**If asked "what's the weakest part?"** the honest answer is: A0 and A2 are measured laws, not
theorems, and the recursion constant (the 2.4×) is not fully derived. Everything in the
"prove" rows is unconditional given the slope bound. Owning this *is* the paper's credibility.

---

## 7. The two-horned trap — the experiment that proves the ODE is right

Fixed `t₀`, no anchor, and toggle the sampler's reverse-SDE noise:
- **noise ON** → stuck fat at 10.8–13.2σ² (the stochastic horn),
- **noise OFF** → collapses to a *single point* within one generation (the deterministic horn).

**Why this is the decisive ablation:** the variance ODE says `−dV/dt = (1−2κ̂/s)V + 1`. The
`+1` is the injected noise. Turn it off and the equation loses its forcing → `V → 0` (point
collapse). Keep it on and the finite ceiling `κ̄` makes the floor *larger* than `σ²` (fat).
The two horns **bracket the true width from both sides**, and no noise level hits it. This
falsified an earlier guess of mine (that noise-off would land on the deterministic floor
`4.5σ²`) — the ODE was right, the guess was wrong. That's the kind of thing to volunteer, not
hide.

---

## 8. The honest gaps and how to defend them (the recent hardening)

Two gaps are openly flagged. You should be able to state each *and* its current defense.

**Gap A — the 2.4× recursion residual.** Iterating the single-pass floor with the clean-tube
ceiling gives `4.5σ²`; the measured recursion floor is `~11σ²`. The theorem only claims a
*lower bound* on the recursion, so this is respected, not a contradiction. **Partial closure:**
the ceiling itself *degrades* on fatter pools — measured `κ̄(w) ≈ 4.4 − 11.6√w`, an
**empirical fit, not a derived law** (say "empirical fit"; the `√w` is cosmetic — five
functional forms including a fit-free interpolation move the answer ≤10%). Feeding it back
self-consistently moves `4.5 → 5.6–7.1σ²` across three seeds, shrinking the factor to
1.5–2.0×. **What the remainder is NOT:** fluctuation (Jensen) bias is 0.006σ² (negligible);
the measured slope-tracking lag moves it +0.3–0.4σ². Leading remaining suspect: the network's
profile *shape* inside the deficit band, which we never measured.

**Gap A's sharpest attack, repelled:** the closure assumes degradation depends on pool *width*
only, but real pools are *mixtures* (clean tube + fattened returns). Direct test: train on the
fixed point's own mixture composition → `κ̄ = 3.37 ± 0.15` vs single-tube prediction `3.34`.
The width-only assumption survives.

**Gap B — the small net response `c`.** Measured `c ≈ +0.05` is a near-cancellation. Anatomy:
deficit channel `≈ −0.94` (fatter pool → easier → less error) + capacity-degradation `≈ +0.7`
+ residual injection `≈ +0.3`. **Be honest that injection is *defined* as the residual**, so
the channels sum to the measurement by construction — the non-trivial content is that the
residual comes out stable, positive, and the right size. Don't claim the sum "validates"
anything; it's a consistency check.

---

## 9. Quiz yourself — questions a professor or reviewer will actually ask

If you can answer all of these without notes, you're ready.

1. **Draw `κ(t)`. Why does it come back down to zero?** (Because at `t→0` there's almost no
   noise left, so `s→0` and the slope `s/γ² → 0`. The *demand* for steepness is only in the
   middle window.)
2. **Why doesn't annealing the truncation fix Case 2?** (Truncation is the additive `s²_g`
   term; the plateau is set by `e²`, which no schedule touches. Case 1 is the special case
   `e² ≡ 0`.)
3. **Your floor assumes an affine score. Real scores are nonlinear — why isn't this fatal?**
   (Orthogonality shield: the nonlinear residual has zero first-order effect on the variance
   and can only *add* it. Affine bound is a true lower bound.)
4. **Why is the floor a function of `ρ` alone?** (The rescaled `t=σ²u, V=σ²W` ODE is `σ`-free.
   Universality is a theorem.)
5. **How is anchoring different from a replay buffer?** (Replay changes `λ` and preserves
   samples; anchoring changes `F_g` and overwrites the majority's fine-scale width. Replay
   dilutes the floor; anchoring removes it.)
6. **Why bidirectional and per-axis?** (Sampler error is signed and anisotropic; a one-sided
   or scalar matcher mis-corrects some axes. Measured: scalar leaves thin-ratio 0.62.)
7. **Why is the fix `d`-independent?** (The floor is set by *local* normal slope; the frame
   errors cancel between pool and reference at first order. Confirmed flat over `d=8→128`.)
8. **Is `κ̄(w) = 4.4 − 11.6√w` a law?** (No — an empirical fit. Form-insensitive; used only as
   an interpolant inside the measured range.)
9. **What's the single most convincing evidence for the floor law?** (The interventional
   test: five training protocols move the measured ceiling `κ̄` from 3.5 to 7.5, and the
   measured floor falls 7.8 → 2.5σ², tracking the predicted `1/κ̄²` ordering and never
   crossing the proven lower bound.)
10. **What would falsify the whole thing?** (If a fixed-capacity sampler on a genuinely thin
    manifold reached `σ²` without anchoring; or if anchoring failed to hold a plateau at some
    `λ > 0`; or if the floor didn't scale as `1/κ̄²`. None did.)

---

*Study order suggestion:* §0 → draw `κ(t)` from §1 → the two assumptions §2 (hardest, know
them cold) → the annealing remark in §3 → the orthogonality shield in §4 → the replay-buffer
distinction in §5 → then the quiz. The math is downstream of these intuitions; if the
intuitions are solid the proofs read as bookkeeping.
