"""Adversarial check on the CIFAR anchored arm: is its sub-baseline off-manifold
distance genuine quality, or mode contraction the precision-like metric can't see?

Three complementary NN metrics (all CPU, chunked, so they don't disturb the GPU run):
  PRECISION  : for each GENERATED sample -> NN dist to real reference.
               (my headline metric; low = samples land near real data; a mode-collapsed
                generator emitting a few perfect reals also scores great -> not enough alone)
  COVERAGE   : for each REAL held-out image -> NN dist to nearest GENERATED sample.
               (low = generator covers the data manifold; HIGH = many real modes left
                uncovered = mode collapse. This is the metric the matcher CANNOT game by
                contracting.)
  DIVERSITY  : for each GENERATED sample -> NN dist to another GENERATED sample.
               (intra-set spread; low = samples clumped together = collapse)

Baselines: real-vs-real (TEST[4000:8000] vs TEST[:4000]) anchors every metric.
Reads saved state npz (latest available generation per arm). Read-only.
"""
import os, sys, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

Z = np.load(os.path.join(BASE, "cifar32.npz"))
TEST = Z["test"]
REF = TEST[:4000]            # reference (same as the run's TESTREF)
HELD = TEST[4000:8000]       # held-out real images, for coverage + real-vs-real baseline

def nn_dist(A, B, exclude_self=False, chunk=256):
    """For each row of A, min L2 distance to any row of B."""
    bn = (B * B).sum(1)
    out = np.empty(len(A), dtype="float64")
    for i in range(0, len(A), chunk):
        c = A[i:i + chunk]
        d2 = (c * c).sum(1)[:, None] + bn[None, :] - 2.0 * c @ B.T
        if exclude_self:                       # A is B: blank the diagonal block
            for j in range(len(c)):
                gi = i + j
                if gi < d2.shape[1]:
                    d2[j, gi] = np.inf
        out[i:i + chunk] = np.sqrt(np.clip(d2.min(1), 0, None))
    return out

def summ(name, d):
    print(f"  {name:26s} mean={d.mean():7.3f}  median={np.median(d):7.3f}  "
          f"p90={np.percentile(d,90):7.3f}")

# real-vs-real baselines
print("=== BASELINE: real vs real (TEST[4000:8000] vs TEST[:4000]) ===")
prec_rr = nn_dist(HELD, REF)
cov_rr = nn_dist(REF, HELD)             # symmetric role for a coverage reference
div_rr = nn_dist(HELD, HELD, exclude_self=True)
summ("precision (real->ref)", prec_rr)
summ("coverage  (ref->real)", cov_rr)
summ("diversity (real intra)", div_rr)

for arm in ["unfixed", "fixed"]:
    p = os.path.join(BASE, "cifar_state", f"{arm}_s{SEED}.npz")
    if not os.path.exists(p):
        print(f"\n[{arm}] no state file yet"); continue
    z = np.load(p); gen = z["cur"]; g = int(z["g"])
    print(f"\n=== {arm.upper()} arm, generation {g}  ({len(gen)} samples) ===")
    prec = nn_dist(gen, REF)            # generated -> real reference (headline metric)
    cov = nn_dist(HELD, gen)            # real held-out -> nearest generated (COVERAGE)
    div = nn_dist(gen, gen, exclude_self=True)
    summ("precision (gen->ref)", prec)
    summ("coverage  (real->gen)", cov)
    summ("diversity (gen intra)", div)
    # verdict signals relative to real-vs-real
    print(f"  --> precision vs real baseline: {prec.mean()/prec_rr.mean():.2f}x "
          f"(<1 = closer to ref than real images are)")
    print(f"  --> COVERAGE vs real baseline:  {cov.mean()/cov_rr.mean():.2f}x "
          f"(>1 = real modes left UNcovered = collapse signal)")
    print(f"  --> DIVERSITY vs real baseline: {div.mean()/div_rr.mean():.2f}x "
          f"(<1 = samples more clumped than real data = collapse signal)")
