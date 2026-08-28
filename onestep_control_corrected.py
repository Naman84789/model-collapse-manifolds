"""
Does the flat control account for the ring's remaining discrepancy?

curved_ode_onestep.py measures, on a frozen ensemble and one paired antithetic Euler step,
the residual of the flat ODE against the exact curvature term (D-1)Cov(r,1/r). On the ring
the ratio is 0.99 where the term is large and degrades as the term shrinks. The flat segment
runs the identical estimator on a geometry where the term is IDENTICALLY ZERO, so whatever
residual it reports is the estimator's own error at that t and step.

This joins the two and asks whether the ring's discrepancy is that same error. If it is, the
identity holds within measurement error at every t, not only where the term is large.

Run:  python onestep_control_corrected.py
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(BASE, "curved_ode_onestep.jsonl"))]
ring = {(round(r["t"], 4), r["dtau"]): r for r in rows if r["geom"] == "ring"}
seg = {(round(r["t"], 4), r["dtau"]): r for r in rows if r["geom"] == "seg"}

for dtau in sorted({k[1] for k in ring}, reverse=True):
    print(f"\n=== dtau = {dtau:g} ===")
    print(f"  {'t':>8} {'ring resid':>11} {'curv term':>11} {'discrepancy':>12}"
          f" {'flat control':>13} {'corrected':>11} {'ratio':>7} {'corr.ratio':>11}")
    for t in sorted({k[0] for k in ring}, reverse=True):
        r = ring.get((t, dtau)); s = seg.get((t, dtau))
        if not r or not s:
            continue
        disc = r["resid"] - r["curv_term"]
        ctrl = s["resid"]                      # true value there is exactly 0
        corr = disc - ctrl                     # discrepancy not explained by the control
        rat = r["resid"] / r["curv_term"] if r["curv_term"] else float("nan")
        crat = (r["resid"] - ctrl) / r["curv_term"] if r["curv_term"] else float("nan")
        print(f"  {t:>8.4f} {r['resid']:>11.5f} {r['curv_term']:>11.5f} {disc:>12.5f}"
              f" {ctrl:>13.5f} {corr:>11.5f} {rat:>7.3f} {crat:>11.3f}")
print("\n'corrected' = ring discrepancy minus the flat control's error at the same t and step.")
print("'corr.ratio' = the same, expressed as a ratio to the curvature term.")
