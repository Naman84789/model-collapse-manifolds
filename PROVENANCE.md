# Claim provenance

Every numeric claim in `paper/main.tex` should trace to a logged run in this repo.
This file records the sweep and, honestly, the claims that do not.

## Swept and verified (2026-09-02)

| claim | source | status |
|---|---|---|
| MNIST unanchored 3.06, anchored 1.66, baseline 1.34 | `pixel_mnist_recursion.jsonl` | exact (final-gen means 3.0587 / 1.6583 / 1.3376) |
| MNIST closes ~81% of the gap | same | exact (1.4004/1.7211 = 81.4%) |
| MNIST ratios 2.3x and 1.24x | same | exact |
| head-to-head annealed 9.8 sigma^2 | `head_to_head.jsonl` | exact (9.840, 5 seeds, 20 gens) |
| head-to-head anchored 1.01 +- 0.02 | same | exact (1.006 +- 0.020) |
| replay 8.5 +- 1.1 | `replay_baseline.jsonl` | exact (8.519 +- 1.122) |
| subcritical control 16.5 +- 2.2 | `subcritical_lambda.jsonl` | exact (16.487 +- 2.249) |
| capacity_domains probe ratios | `capacity_domains_samplerlaw.jsonl` | exact (0.995-0.998, 0.740-0.746, 0.975-0.988) |
| kbar sampler/pt ratio 0.83-0.87, monotone | `kbar_sampler_law.jsonl` | exact (0.827-0.868, monotone) |
| poolwidth fits 3.80-10.25 sqrt(w), 4.54-12.17 sqrt(w) | `poolwidth_probe_samplerlaw.jsonl` | exact |
| geometry drops 42% / 30% / 49% | `capacity_domains.jsonl` | exact; probe-insensitive (sampler law gives 42.2 / 29.2 / 48.0) |
| truncation remainder ~39 sigma^2 | analytic | exact: (1-e^-0.05)/0.5/sigma^2 = 39.0 |

## Corrected during the sweep

| claim | was | now | why |
|---|---|---|---|
| CIFAR off-manifold | 22.8 +- 1.2 / 16.3 +- 0.7, dev +2.7 / -3.8 | 22.5 +- 0.5 / 16.3 +- 0.4, dev +2.4 / -3.9 | neither value appears in `CIFAR_3SEED_RESULTS.txt`; logged run is 22.52 +- 0.49 and 16.25 +- 0.41 against a 20.15 baseline |
| capacity degradation range | 3.7-4.3 down to 2.8-3.0 | 3.2-3.6 down to 2.4-2.6 under eq. (3), 4.0-4.4 down to 2.8-3.1 under p_t | no jsonl contained the quoted ranges |
| head-to-head annealed s.d. | +- 1.7 | +- 1.8 | 1.7 is ddof=0; every other error bar in the paper is ddof=1 (1.846) |
| subcritical s.d. | 69 +- 16 | 69 +- 15 | ddof=1 gives 15.475 |
| interventional ceilings | 3.5 -> 7.5 (p_t) | 3.2 -> 5.8 (eq. 3) | figure plotted p_t circles beside a sampler-law star |
| profile-sag channel | sag 0.59-0.68, worth 0.54-0.88 sigma^2 | sag 0.50-0.56, worth 1.09-1.37 sigma^2 | computed from the p_t profile inside a paragraph written under eq. (3) |

## NOT verified: no logged source in this repo

- **"21 vs 68 sigma^2 in raw normal variance"** (jump vs standard unanchored plateau at
  lambda=1/2, matched truncation t_0 = t_hat = 0.05), in the attribution paragraph of
  Section 7. The accompanying analytic figure (~39 sigma^2 truncation remainder) checks
  out, and the ratio 68/21 = 3.24 is consistent with the stated "~3x", but neither
  plateau value appears in any jsonl or json file here. It appears to come from a run
  that was never logged. Either re-run and log it, or soften the claim.

## Ring floor anatomy and appendix (swept 2026-09-02)

| claim | source | status |
|---|---|---|
| four-channel budget 2.6 / 3.0 / 7.2 / 7.3 % | `gap_budget.jsonl` | exact; each recomputes from the stage values |
| budget product 2.90 vs measured 2.894 | same | exact (3.564 x .974 x .970 x .928 x .927 = 2.897) |
| seg ensemble width 1.06-1.43x, skew +0.05 to -0.03 | `ensemble_shape.jsonl` | exact |
| seg predicted probe ratio 0.996 vs 0.995-0.998 | same | exact |
| ring width 2.05x, skew -0.44 to -1.02, offset 0.03-0.04 | same | exact (skew -0.436 to -1.023) |
| sigma->0 constant (1+3e^-4)/8 = 0.1319 | closed form | exact; `falsify_floor.py` T2 gives g*rho^2 = 0.527474 vs C_0 = 0.527473 |
| "1/8 and (1+3e^-4)/8 differ by 5.5%" | arithmetic | exact (0.13187/0.125 = 1.055) |
| rho=0.39: 3.953/3.812/3.775 vs limit 3.769 | `falsify_floor.py` T5 | exact, reproduced by running it |
| `falsify_floor.py` adversarial battery | ran it | **14/14 pass** |
| `e2e_floor_check.py` end-to-end battery | ran it | **7/7 pass** |
| note_floor: "recovers 1/(2 rho^2) to 1.7%" | `e2e_floor_check.py` test E | exact (+1.70 / +1.65 / +1.73 % at rho = 0.2 / 0.39 / 0.55) |

### Corrected

| claim | was | now | why |
|---|---|---|---|
| skew injection endpoint | 3.47 -> 3.30 | 3.47 -> 3.37 | `gap_budget.jsonl` records `three_moment_matched: 3.367`; a move to 3.30 would make the skew channel 4.9%, contradicting the 3.0% the paper uses two sentences later |

### Noted, not an error

`deficit_floor_law.py`'s docstring calls 0.141 the "strongly-singular asymptotic". It is
not the asymptote; it is the value at finite rho. Since Phi*kbar^2 = (rho^2/4) g(rho) =
C_0/4 + O(rho^2) = 0.1319 + O(rho^2), and the logged points sit at rho = 2 sigma kbar =
0.15 to 0.30, the O(rho^2) term accounts for the difference. The paper itself only ever
states the exact 0.1319 and never claims the numerics equal it, so no paper claim is
affected; the docstring is merely loose.

## Enforced automatically (from 2026-09-02)

`provenance_check.py` now checks 19 load-bearing numbers by pairing a regex that extracts
the value AS PRINTED IN THE PAPER with a function that recomputes it FROM THE LOGGED DATA.
Re-run any experiment and the dependent checks fail until the text is updated. Run it
before any commit that touches numbers:

    python provenance_check.py        # exit 0 = pass, 1 = at least one mismatch
    python provenance_check.py -v     # show passing rows too

Tolerance is set by the printed precision, not relatively: a paper printing "1.1" asserts
the source rounds to 1.1, so the check is |source - 1.1| <= 0.05. That is the correct
semantics for a rounded display and it is what caught the remaining CIFAR error.

### Found by the checker on its first run

**`CIFAR_3SEED_RESULTS.txt` disagrees with itself.** Three of its stated per-seed means do
not match the mean of the per-generation values listed directly above them:

| | stated | recomputed from its own listed generations |
|---|---|---|
| UNFIXED seed 2 | 22.49 | 22.3688 |
| FIXED seed 1 | 15.82 | 15.8950 |
| FIXED seed 2 | 16.66 | 16.5375 |

The overall means inherit it: 22.52 stated against 22.4788, and 16.25 against 16.2358.
The paper had taken its figures from the summary half. It now takes them from the
per-generation values, which are the finest-grained data present, using mean and s.d. over
the three seed means as the table caption promises: unanchored 22.5 +- 0.5 (deviation
+2.3), anchored 16.2 +- 0.3 (deviation -3.9). The earlier hand correction had reached
22.5 +- 0.5 correctly but left the anchored arm at 16.3 +- 0.4.

The cause is not established and resolving it needs a re-run, so the checker reports it as
a standing WARNING rather than a failure: no paper claim depends on the inconsistent half,
and a permanently red check is one people learn to ignore.
