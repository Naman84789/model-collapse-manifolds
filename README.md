# Manifold model-collapse — reproducibility packet

Self-consuming (recursive) training of diffusion models on **manifold-supported
(singular) data** — the σ→0 / infinite-Fisher regime that 2606.13796 (Cambridge) left
as an explicitly **open Case 2**. This directory contains the code, data, and figures
behind the result: annealed truncation (Cambridge's remedy) cannot reach the data
distribution in this regime, and a data-anchoring matcher does.

**Read [`PROOFS.md`](PROOFS.md) for the theorems and the honest ledger (§5).** Claims are
tiered there as **proved** (theorems / lower bounds), **experimentally validated**
(the runs below), and **partially modeled** (empirical mechanisms). This README is the
run-order map; it does not restate the math.

This repository is a curated subset of a larger private research directory: it contains
everything needed to reproduce the paper's claims (scripts, logs, the paper itself), but
two sections below (*Supporting / earlier-phase evidence*, *Not load-bearing*) describe
earlier or superseded scripts that live only in that private directory, kept here for
context — they are not part of this repository.

---

## Environment

- **CPU-only** for the core pipeline (steps 1–13; all runs were done on 12 threads, no
  GPU needed). The CIFAR-10 scale check (steps 14–16) is GPU-only in practice (small conv
  UNet, AMP; a laptop RTX 3050 took ~4.75 h/seed) — it is a preliminary supplement, not a
  load-bearing claim of the paper.
- Python 3.11+, with `numpy`, `torch`, `matplotlib`. `torchvision` is only
  needed once to build the MNIST cache (a prebuilt `mnist8x8.npz` is already included).
  `falsify_floor.py` additionally needs `scipy` (and uses `sympy` if installed). The CIFAR
  scripts need `pyarrow` and `pillow` to decode the HuggingFace parquet.
- Each script sets `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` before importing torch and
  calls `torch.set_num_threads(12)` — no shell setup required.
- Every long run is **append-only resumable**: results stream to a `*.jsonl`; re-running a
  script skips keys already present. Kill and restart freely.

Run anything with `python <script>.py`.

---

## Core pipeline (this is the paper)

Ordered as the argument is built. "Establishes" ties each run to a claim; "→ figure" ties
it to a panel in [`paper/figs/`](paper/figs) (the committed, final figures). Running
`make_figures.py` yourself regenerates them into a local `figures/` first, exactly as in
step 12 below.

| # | Script | Output | Establishes | → figure | ~time |
|---|--------|--------|-------------|----------|-------|
| 1 | `rprime_probe.py`, `rprime_probe2.py` | `rprime_probe25.jsonl`, `rprime_probe2_25.jsonl`, `rprime_probe_net25.pt` | **Measured** network normal-slope ceiling κ̄ ≈ 3.9, residual field r²≈0.03, ν≈0.75 (R=2.5 ring). The single input the floor prediction needs. | — | ~3 min |
| 2 | `deficit_floor_law.py` | `deficit_floor_law.jsonl` | Finite-σ floor table Φ_det(κ̄): Φ_det/σ²={14.1,6.28,3.82,2.44,1.43,1.00} at κ̄={2,3,3.9,5,7,10} (σ=0.05). κ̄=3.9→3.8σ² matches the measured single-pass floor with **zero fit**. (Thm I(a), items 1 & 6) | **fig4** | seconds |
| 2b | `floor_theory_upgrade.py` + `closed_form_check.py` | stdout (tables) | **CLOSED FORMS (proved).** Limit ODE in ρ=2σκ̄ alone (universality = theorem, Lemma 3.2); no-overshoot g(ρ) explicit, C₀=(1+3e⁻⁴)/2, Φ→0.1319/κ̄²; **unconditional** (κ̂≤κ̄ only) g=1/(2ρ²) exact, Φ→1/(8κ̄²), floor>σ² ⟺ ρ<1/√2. Closed forms vs numerics ≤2.4e-5. "0.141" = finite-σ effective value of exact 0.1319. (Lemmas 3.2/3.3, challenge #13) | — | ~4 min |
| 3 | `head_to_head.py` | `head_to_head.jsonl` (15) | **Headline.** Cambridge annealed truncation (t₀:0.05→1e-4, no anchor) vs ours (std+anchor, jump+anchor). 5 seeds, G=20, λ=½. Arm A → 9.84σ²; Arms B,C → 1.01σ², flat. (items 2 & 3) | **fig1** | ~59 min |
| 4 | `analyze_head_to_head.py` | `head_to_head_figure.json` (stdout) | Mean±std trajectories + the deterministic recursion fixed point v*_det for overlay. | (fig1) | seconds |
| 5 | `inject_ablation.py` | `inject_ablation.jsonl` (6) | **Break-test / two-horned trap.** Fixed t₀=1e-4, no anchor, reverse-SDE noise ON vs ZEROED. ON → 10.8–13.2σ² (fat); OFF → 0. (challenge #12) | **fig2** | ~22 min |
| 6 | `verify_nonoise.py` | stdout | Confirms the noise-OFF collapse is a **single point** (2D std→4e-4, NaN-free), not an artifact. | (fig2) | ~1 min |
| 7 | `det_fp_diagnose.py` | stdout | The honest gap: Φ_det recursion fixed point = 4.54σ² at the method's real t₀; measured stochastic recursion ≈11σ² is 2.4× above → Φ_det is an exact single-pass floor and a **lower bound** on the recursion. (Half of the 2.4× is now accounted for — see row 17.) | — | ~2 min |
| 8 | `pixel_mnist_recursion.py` | `pixel_mnist_recursion.jsonl` (8) | **Real-data step.** 8×8 pixel MNIST, G=8, 3 seeds. Unfixed → off-manifold 3.06; anchored → 1.66 (true baseline 1.34). Needs `mnist8x8.npz`. (item 4) | **fig3** | ~30 min |
| 9 | `poolwidth_probe.py` | `poolwidth_probe.jsonl` (6) | **Small-c resolved.** Net c≈{0,.05,.13,.10,.05} = deficit response −0.68 + injection response +0.73 (near-cancellation). (item 5) | — | ~5 min |
| 10 | `jump_dscaling.py` | `jump_dscaling.jsonl` (12) | Supporting: raw floor blows up ~linearly in ambient d; matched floor **saturates** (~d^0.1); no compounding to d=128. | — | ~20 min |
| 11 | `test_axmatch.py` | stdout | Unit test of the pixel-space local matcher (no crash when pool>ref; off-manifold 3.02→1.55). | — | ~1 min |
| 12 | `make_figures.py` | `figures/fig1–4.png` | Regenerates all four paper figures from the jsonl above. | all | seconds |
| 13 | `falsify_floor.py` | stdout (14 checks) | **Adversarial verification of Theorem 2.** Independent RK45 integration vs both closed forms; sympy symbolic re-derivation of the band crossing; limits ρ→0, ρ→1⁻, κ̄→∞, λ→1; boundary-layer contraction at the predicted quadratic rate; 200-profile Monte Carlo attack on the comparison bound (incl. overshoot). 14/14 pass. | — | ~1 min |
| 14 | `cifar_recursion.py` | `cifar_state/*.npz` (not committed, large), `cifar_run_offman.log` | **Scale check (preliminary).** Same recursion at D=3,072 (CIFAR-10 pixels), small conv UNet (~7M params, GPU/AMP), G=8, λ=½, 3 seeds. Unanchored settles at 22.8±1.2σ (true baseline 20.1); anchored holds 16.3±0.7, gap +6.5±1.1 in every seed. Needs a CUDA GPU in practice (~4.75 h/seed on an RTX 3050); downloads CIFAR-10 from HF parquet on first run (script prints the fetch command). | **fig6_cifar** | ~4.75 h/seed |
| 15 | `cifar_diversity_check.py <seed>` | stdout | **Adversarial check on 14: is the anchored arm's lower off-manifold distance real quality or mode collapse the distance metric can't see?** Three NN metrics vs a real-vs-real baseline: precision (0.80×), **coverage (1.00×, rules out mode collapse)**, diversity (0.75×, bounded/non-compounding — identical at g4 and g7). Unanchored: precision 1.11×, coverage 1.16× (real modes go uncovered), diversity 1.20×. Read-only, CPU, <1 min per seed. | (fig6_cifar) | ~1 min |
| 16 | `analyze_seeds.py` | stdout | Cross-seed trend analysis: unanchored slope +0.238±0.259/gen (noisy, not monotone — seed 2 is flat, so this is a stable degraded fixed point at λ=½, not runaway collapse); anchored slope −0.073±0.141/gen (flat). Full numbers in `CIFAR_3SEED_RESULTS.txt`. | — | seconds |
| 17 | `gap_closure.py` | stdout | **The 2.4× gap, a-quarter-to-half closed.** The width probe's own per-width slope fits give an *empirical* capacity-degradation fit κ̄(w) ≈ 4.4 − 11.6·√w (seed-501 nets; seeds 500/777 give 4.06 − 9.7·√w and 4.68 − 13.0·√w — rows 19–20 measure the scatter and replication). Feeding it back self-consistently, v ← Φ_det(w(v), κ̄(w(v))), moves the deterministic fixed point 4.54σ² → 5.57–7.13σ² across the three seeds' fits at t₀=1e-4, and the resulting √w\* = 0.091–0.101 sits **inside** the measured support [0.03, 0.13] (interpolation, not extrapolation). Shrinks the unexplained factor 2.4 → 1.5–2.0 (a quarter to a half of the log gap); for the rest see row 20 (what it is *not*). Pure numerics on logged data. | — | ~2 min |
| 18 | `gap_b_decompose.py` | stdout | **Three-channel anatomy of c.** Splits dΦ_det(w, κ̄(w))/dw into deficit (∂/∂w at fixed κ̄ ≈ −0.94 mid-range) + capacity degradation ((∂Φ/∂κ̄)·κ̄′(w) ≈ +0.7) + residual injection (≈ +0.3 to +0.4). Injection is *defined* as measured-minus-derived, so the three channels sum to the measurement by construction; the non-trivial content is that the residual comes out stable, positive, and of the size the residual-channel model expects — and that most of what row 9 attributes to "injection" is really κ̄ degrading on fatter pools. Trust mid-range only: κ̄′(w) diverges at the thinnest width (a fit artifact of the √w parametrization). | — | ~3 min |
| 19 | `gap_review_checks.py`, `gap_review_torch.py` | stdout | **Adversarial review battery on rows 17–18** (same spirit as row 13). R1 functional form: refitting κ̄(w) linear/log/quadratic or replacing it with a fit-free monotone interpolation moves v\* by ≤10% (PCHIP lands on the identical 6.11σ²) — the √w is cosmetic, the closure only interpolates. R2 honest closure arithmetic (the source of row 17's closure fractions). R3 the capacity-degradation channel recomputed from raw centered differences, no functional form: +0.78/+0.68/+0.67 vs fit +0.84/+0.70/+0.65 at the interior widths. R4 κ̄ seed scatter: level shifts ~0.3 between seeds (→ v\* 6.11–7.13σ²), slope consistent. R5 **mixture test**: the recursion pool is a mixture, not a single tube — training on the fixed point's own composition (half σ=0.05, half σ=0.124) gives κ̄ = 3.37 ± 0.15 vs single-tube prediction 3.34; the width-only reduction survives its sharpest objection. | — | ~10 min |
| 20 | `gap_protocol_robustness.py`, `gap_residual_attack.py` | stdout | **Replication + residual forensics.** Replication: the κ̄ degradation replicates under a half-width network (law 4.25 − 9.4·√w), a different ring radius R=3.5 (4.29 − 11.2·√w), and a third seed (4.68 − 13.0·√w) — the *phenomenon* is protocol-robust (slope −9.4 to −13.0, κ̄ falls ≈3.9–4.3 → ≈2.9–3.0 everywhere); the *constants* are protocol-specific. Residual forensics — what the remaining 1.5–2.0× is **not**: (a) Jensen/fluctuation bias of iterating a noisy map, computed from the logged fixed-t₀ plateau variance: 0.006σ², negligible; (b) the measured t≥0.1 slope-tracking ratio 0.95–0.97 applied to the whole profile: moves v\* by only +0.3–0.4σ². Leading remaining suspect: the profile shape inside the deficit band, where tracking is unmeasured. | — | ~3 min |

### Regenerate the four figures
```
python make_figures.py     # reads head_to_head.jsonl, inject_ablation.jsonl,
                           # pixel_mnist_recursion.jsonl; writes figures/fig1–4.png
```
All inputs are already present, so this runs in seconds without redoing any training.

---

## Headline numbers (check a correct reproduction against these)

σ² = 0.0025 throughout (σ = 0.05).

- **Head-to-head (fig1):** Arm A g19 = 0.0246 ≈ 9.84σ²; Arms B,C = 0.00252 ≈ 1.01σ², flat 20 gens.
- **Two-horned trap (fig2):** stochastic ≈ 0.027–0.033 (≈11σ²); deterministic → 0 (point collapse).
- **Pixel MNIST (fig3):** unfixed g7 = 3.06; anchored g7 = 1.66; true-data baseline = 1.34.
- **Capacity floor (fig4):** finite-σ Φ_det·κ̄² = 0.141 ± 0.003 (σ=0.05); exact σ→0 constants: (1+3e⁻⁴)/8 = 0.1319 (no-overshoot), 1/8 (unconditional); measured κ̄=3.9 → 3.8σ² (zero-fit).
- **Honest gap, a quarter-to-half closed:** recursion Φ_det fixed point 4.54σ² vs measured ≈11σ² → factor 2.4. Self-consistent iteration with the empirical capacity fit κ̄(w) ≈ 4.4 − 11.6·√w gives v\* = 5.57–7.13σ² across three probe seeds (t₀=1e-4), √w\* = 0.091–0.101 inside the measured support, form-insensitive (≤10% across five fits incl. fit-free interpolation); mixture test κ̄ 3.37±0.15 vs 3.34 predicted; degradation replicates at h=128 and R=3.5. Residual factor 1.5–2.0 stays open — but Jensen bias (0.006σ²) and the measured tracking lag (+0.3–0.4σ²) are excluded as its source.
- **Three-channel c:** deficit ≈ −0.94 + capacity degradation ≈ +0.7 (form-free check: +0.78/+0.68/+0.67) leaves residual injection ≈ +0.3–0.4, stable and positive (residual is defined as measured-minus-derived — a plausibility check, not an independent validation).
- **CIFAR-10 pixels, D=3,072 (fig6_cifar, preliminary, 3 seeds):** unanchored 22.8±1.2 (true baseline 20.1); anchored 16.3±0.7; gap +6.5±1.1, same sign every seed. Anchored coverage 1.00× (no mode collapse), diversity 0.75× (bounded). Samples are blurry at this model budget — this is a dynamics/coverage check, not a perceptual-quality claim (see `CIFAR_3SEED_RESULTS.txt`).

---

## Supporting / earlier-phase evidence (not included in this repo)

Theorem II′ (the collapse law) and the fix were first established in an earlier,
2026-07-02 solution phase, ahead of the polished pipeline above. Those scripts
(`solution_law_test.py`, `solution_validation.py`, `scale_experiment.py`,
`scale_variants.py`, `jump_solution_test.py`, `grand_harness.py`,
`manifold_fix_test.py`/`_test2.py`) underpin the results narratively but are superseded by
the core pipeline's cleaner re-derivation and are **not part of this repository** — they
live only in the author's private research directory. `SOLUTION_NOTE.md` and
`THEORY_NOTE.md` (included here) narrate that phase and carry in-place correction banners
where later results superseded earlier claims; the corrected, citable statements are in
`PROOFS.md`.

**Doc hygiene (challenge #11) — DONE 2026-07-04:** the superseded "jump breaks the floor" /
"jump sits 6× below Φ(0+)" lines in `SOLUTION_NOTE.md §4c` and `THEORY_NOTE.md` now carry
in-place **[CORRECTION 2026-07-04]** banners (the *matcher* is the floor-breaker; the jump
is a raw-error reducer). Safe to quote those files — the banners travel with the prose.
Corrected statements live in `PROOFS.md` Theorems I(b)/III(b).

---

## Not load-bearing for this paper

Left in the directory but **not part of the manifold-collapse result** — exploratory
probes, other topics, or superseded lines. Ignore for reproduction:

- `probe_ragas_*.py`, `probe_gsm8k_extract.py` — LLM-faithfulness probes (different topic).
- `minority_extinction_law.py`, `real_model_minority.py`, `gmm_crossover.py`, `crossover.py`,
  `collapse_gaussian_repro.py`, `collapse_vae_*.py`, `collapse_phase_*.py`,
  `collapse_break_accumulate.py`, `complexity_bias.py`, `entropy_headtohead.py` — earlier
  angles / non-manifold collapse explorations.
- `heavytail_a4_probe.py`, `tail_vs_poison_*.py`, `annealed_fix_heavytail.py` — the
  heavy-tail critique line (superseded: the A4-vs-Fisher distinction resolved it; see the
  `cambridge-spectral-collapse` note).
- `diffusion_collapse_fulltest.py` (+ `analyze_fulltest.py`), `problem_verify_*.py`,
  `loop_verifier_center.py` — scaffolding / one-off verifications.

---

## Data artifacts

- `mnist8x8.npz` — 8×8 pixel MNIST cache (train/test, [-1,1]) for `pixel_mnist_recursion.py`.
- `mnist_latents32.npz` — 32-d PCA latents (used by earlier scale experiments).
- `rprime_probe_net25.pt`, `rprime_probe_net.pt` — trained score nets from the κ̄ probe.
- `cifar32.npz` (737 MB, **not committed**) and `cifar_state/*.npz` (73 MB × 5, **not
  committed**) — `cifar_recursion.py` fetches/builds and caches these on first run;
  `cifar_run_offman.log` (committed) has the full per-generation trajectory, and
  `CIFAR_3SEED_RESULTS.txt` has the final numbers if you just want to check without
  re-running.
