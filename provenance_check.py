"""
Check every load-bearing number in paper/main.tex against the data that produced it.

WHY THIS EXISTS. A hand audit on 2026-09-02 found seven defects in the paper, all of one
species: numbers that outlived the run that produced them. CIFAR figures carried over from
an earlier run; a ceiling range present in no jsonl at all; forward-process values quoted
inside sampler-law arguments; a skew endpoint contradicting the percentage stated two
sentences later; two error bars compared in one sentence computed with different ddof.
Every one was findable only by hand, and every one would reappear the moment a script is
re-run.

WHAT THIS DOES. Each check pairs a regex that extracts a number AS PRINTED IN THE PAPER
with a function that recomputes it FROM THE LOGGED DATA. A mismatch fails. Re-run any
experiment and the checks depending on it fail until the text is updated to match.

This is deliberately a registry rather than anything clever. Mapping prose to provenance
cannot be automated, but it can be written down once and then enforced forever. Adding a
claim to the paper means adding a line here.

Run:  python provenance_check.py        exit 0 = all pass, 1 = at least one failure
      python provenance_check.py -v     also print the passing rows
"""
import json
import math
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(BASE, "paper", "main.tex")
S2 = 0.05 ** 2


# ----------------------------------------------------------------- data helpers
def jl(name):
    rows = []
    for line in open(os.path.join(BASE, name), encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def keyed(name):
    return {r["key"]: r for r in jl(name) if "key" in r}


def mean_sd(xs, ddof=1):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (n - ddof)) ** 0.5


def finals(rows, prefix, scale=1.0):
    return [r["traj"][-1] * scale for r in rows
            if r.get("key", "").startswith(prefix) and "traj" in r]


def cifar(arm):
    """CIFAR off-manifold values live in a text report, not a jsonl."""
    txt = open(os.path.join(BASE, "CIFAR_3SEED_RESULTS.txt"), encoding="utf-8").read()
    block = txt.split("ACROSS-SEEDS")[0]
    vals = []
    # "FIXED" also matches inside "UNFIXED"; the lookbehind stops that.
    for m in re.finditer(r"(?<![A-Z])" + arm + r":\s+((?:g\d+=[\d.]+,?\s*)+)", block):
        vals.append([float(x) for x in re.findall(r"g\d+=([\d.]+)", m.group(1))])
    if len(vals) != 3:
        raise AssertionError("expected 3 CIFAR %s seeds, got %d" % (arm, len(vals)))
    means = [sum(v) / len(v) for v in vals]
    return list(mean_sd(means))


# ----------------------------------------------------------------- the registry
CHECKS = []


def check(name, pattern, tol=0.0, warn=False):
    """warn=True: a known, documented discrepancy that no paper claim depends on.
    Reported every run, but does not fail the build. A permanently red check gets
    ignored, which defeats the point; a warning stays visible."""
    def deco(fn):
        CHECKS.append((name, pattern, fn, tol, warn))
        return fn
    return deco


@check("MNIST unanchored final", r"Pixel MNIST[^\n]*?unanchored & \$([\d.]+)\$")
def _mnist_un():
    return [mean_sd(finals(jl("pixel_mnist_recursion.jsonl"), "unfixed_"))[0]]


@check("MNIST anchored final", r"& anchored & \$(1\.\d+)\$ & 3")
def _mnist_fx():
    return [mean_sd(finals(jl("pixel_mnist_recursion.jsonl"), "fixed_"))[0]]


@check("MNIST true baseline", r"& true-data baseline & \$(1\.\d+)\$ & n/a")
def _mnist_true():
    return [keyed("pixel_mnist_recursion.jsonl")["TRUE_baseline"]["offman"]]


@check("MNIST gap closed pct", r"closing \$\\sim\$(\d+)\\% of the gap", tol=1.0)
def _mnist_gap():
    rows = jl("pixel_mnist_recursion.jsonl")
    u = mean_sd(finals(rows, "unfixed_"))[0]
    f = mean_sd(finals(rows, "fixed_"))[0]
    t = keyed("pixel_mnist_recursion.jsonl")["TRUE_baseline"]["offman"]
    return [round(100 * (u - f) / (u - t))]


@check("head-to-head annealed", r"annealed truncation \\citep\{[^}]*\} & \$([\d.]+)\\pm([\d.]+)\$")
def _h2h_a():
    return list(mean_sd(finals(jl("head_to_head.jsonl"), "A_", 1 / S2)))


@check("head-to-head anchored", r"holds \$([\d.]+)\\pm([\d.]+)\\,\\sigsq\$")
def _h2h_b():
    return list(mean_sd(finals(jl("head_to_head.jsonl"), "B_", 1 / S2)))


@check("replay plateau", r"\$([\d.]+)\\pm([\d.]+)\\,\\sigsq\$ \(5 seeds, same annealed")
def _replay():
    return list(mean_sd([r["traj"][-1] / S2 for r in jl("replay_baseline.jsonl")]))


@check("subcritical control", r"control plateaus at \$([\d.]+)\\pm([\d.]+)\\,\\sigsq\$")
def _sub_ctrl():
    return list(mean_sd(finals(jl("subcritical_lambda.jsonl"), "L25", 1 / S2)))


@check("subcritical divergent arm", r"\(subcritical\) & \$(\d+)\\pm(\d+)\$")
def _sub_div():
    m, s = mean_sd(finals(jl("subcritical_lambda.jsonl"), "L02", 1 / S2))
    return [round(m), round(s)]


@check("skew injection endpoint", r"from \$3\.47\$ to \$([\d.]+)\$")
def _skew():
    return [keyed("gap_budget.jsonl")["BUDGET_ring_t0.005"]["stages"]["three_moment_matched"]]


@check("interventional ceiling range", r"move the ceiling \$([\d.]+)\\to([\d.]+)\$")
def _interv():
    ks = [r["kbar_sampler"] for r in jl("ceiling_origin_samplerlaw.jsonl")]
    return [min(ks), max(ks)]


@check("poolwidth degradation fits",
       r"\\kb\(w\)\\approx([\d.]+)-([\d.]+)\\sqrt w\$ against \$([\d.]+)-([\d.]+)\\sqrt w\$",
       tol=0.02)
def _poolfit():
    f = keyed("poolwidth_probe_samplerlaw.jsonl")["FIT"]["fits"]
    return [f["kbar_sampler"][0], -f["kbar_sampler"][1],
            f["kbar_pt"][0], -f["kbar_pt"][1]]


@check("CIFAR unanchored", r"off-manifold distance \$([\d.]+)\\pm([\d.]+)\$")
def _cif_un():
    return cifar("UNFIXED")


@check("CIFAR anchored", r"it \(\$([\d.]+)\\pm([\d.]+)\$, a deviation of \$-")
def _cif_fx():
    return cifar("FIXED")


@check("sigma->0 constant", r"=([\d.]+)\\,\\kb\^\{-2\}\$", tol=0.0005)
def _c0():
    return [(1 + 3 * math.exp(-4)) / 8]


# --- internal consistency: no tex anchor, the data must agree with itself ---
@check("gap budget channels self-consistent", None)
def _budget():
    b = keyed("gap_budget.jsonl")["BUDGET_ring_t0.005"]
    st, c = b["stages"], b["components_pct"]
    order = ["pt_width_synth", "sampler_width_synth", "three_moment_matched",
             "real_radii_angles_shuffled", "actual_sampler"]
    got = [round(100 * (1 - st[order[i + 1]] / st[order[i]]), 1) for i in range(4)]
    want = [c["width"], c["skew"], c["higher_radial_moments"], c["radius_angle_coupling"]]
    if got != want:
        raise AssertionError("components_pct %s != recomputed %s" % (want, got))
    prod = st["pt_width_synth"]
    for g in got:
        prod *= (1 - g / 100)
    if abs(prod - b["measured"]) > 0.02:
        raise AssertionError("product %.4f vs measured %.4f" % (prod, b["measured"]))
    return None


@check("kbar ratio monotone in sigma", None)
def _mono():
    rs = [r["ratio"] for r in jl("kbar_sampler_law.jsonl")]
    if any(rs[i] >= rs[i + 1] for i in range(len(rs) - 1)):
        raise AssertionError("ratio not monotone: %s" % rs)
    if not (0.82 < min(rs) and max(rs) < 0.87):
        raise AssertionError("ratio band moved: %.4f to %.4f" % (min(rs), max(rs)))
    return None


@check("error bars all use ddof=1", None)
def _ddof():
    """The table caption promises 'mean +- s.d.'. Every bar must be the sample s.d.,
    not the population one. This is the defect that put +-1.7 next to +-1.1."""
    tex = open(TEX, encoding="utf-8").read()
    checks = [
        (finals(jl("head_to_head.jsonl"), "A_", 1 / S2),
         r"annealed truncation \\citep\{[^}]*\} & \$[\d.]+\\pm([\d.]+)\$"),
        ([r["traj"][-1] / S2 for r in jl("replay_baseline.jsonl")],
         r"\$[\d.]+\\pm([\d.]+)\\,\\sigsq\$ \(5 seeds, same annealed"),
    ]
    for xs, pat in checks:
        m = re.search(pat, tex)
        if not m:
            raise AssertionError("could not locate an error bar for the ddof check")
        shown = float(m.group(1))
        s1 = mean_sd(xs, ddof=1)[1]
        s0 = mean_sd(xs, ddof=0)[1]
        if abs(shown - s1) > 0.06 and abs(shown - s0) <= 0.06:
            raise AssertionError(
                "error bar %.2f is the population s.d. (%.3f); sample s.d. is %.3f"
                % (shown, s0, s1))
    return None


@check("CIFAR report self-consistent", None, warn=True)
def _cifar_selfconsistent():
    """The summary block of CIFAR_3SEED_RESULTS.txt must agree with the per-generation
    values it lists above it. On 2026-09-02 it did not: three per-seed means were stated
    up to 0.12 away from the mean of their own listed generations, and the paper had taken
    its figures from the summary half."""
    txt = open(os.path.join(BASE, "CIFAR_3SEED_RESULTS.txt"), encoding="utf-8").read()
    block = txt.split("ACROSS-SEEDS")[0]
    bad = []
    for arm in ("UNFIXED", "FIXED"):
        pat = (r"(?<![A-Z])" + arm + r":\s+((?:g\d+=[\d.]+,?\s*)+)"
               r"\s*mean [^=]*=\s*([\d.]+)")
        for m in re.finditer(pat, block):
            gens = [float(x) for x in re.findall(r"g\d+=([\d.]+)", m.group(1))]
            stated = float(m.group(2))
            actual = sum(gens) / len(gens)
            if abs(stated - actual) > 0.05:
                bad.append("%s stated %.2f vs listed %.4f" % (arm, stated, actual))
    if bad:
        raise AssertionError("; ".join(bad))
    return None


# ----------------------------------------------------------------- runner
def main():
    verbose = "-v" in sys.argv
    tex = open(TEX, encoding="utf-8").read()
    fails = []
    print("%-36s %20s %20s  verdict" % ("check", "in paper", "from data"))
    print("-" * 92)
    warns = []
    for name, pat, fn, tol, warn in CHECKS:
        if pat is None:                       # self-consistency, no tex anchor
            try:
                fn()
                if verbose:
                    print("%-36s %20s %20s  PASS" % (name, "(internal)", "consistent"))
            except AssertionError as e:
                if warn:
                    warns.append((name, str(e)))
                    print("%-36s %20s %20s  WARN  %s" % (name, "-", "-", e))
                else:
                    fails.append((name, "internal", str(e)))
                    print("%-36s %20s %20s  FAIL  %s" % (name, "-", "-", e))
            continue
        want = fn()
        m = re.search(pat, tex)
        if not m:
            fails.append((name, "REGEX MATCHED NOTHING", ""))
            print("%-36s %20s %20s  FAIL  regex matched nothing" % (name, "-", "-"))
            continue
        shown = list(m.groups())
        got = [float(g) for g in shown]
        # A printed "1.1" asserts the true value rounds to 1.1, i.e. |true-1.1| <= 0.05.
        # Relative tolerance is the wrong model for a rounded display.
        def half_ulp(txt):
            d = len(txt.split(".")[1]) if "." in txt else 0
            return 0.5 * 10 ** (-d) + 1e-9
        bad = len(got) != len(want) or any(
            abs(a - b) > max(half_ulp(t), tol * abs(b)) for a, b, t in zip(got, want, shown))
        gs = ", ".join("%g" % x for x in got)
        ws = ", ".join("%.4g" % x for x in want)
        if bad:
            fails.append((name, gs, ws))
            print("%-36s %20s %20s  FAIL" % (name, gs, ws))
        elif verbose:
            print("%-36s %20s %20s  PASS" % (name, gs, ws))
    print("-" * 92)
    for n, msg in warns:
        print("WARNING  %s: %s" % (n, msg))
        print("         Known and documented in PROVENANCE.md. No paper claim depends on")
        print("         the inconsistent half; the text uses the per-generation values.")
    if fails:
        print("%d of %d checks FAILED:" % (len(fails), len(CHECKS)))
        for n, g, w in fails:
            print("  %s: paper [%s] vs data [%s]" % (n, g, w))
        return 1
    print("all %d checks pass" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
