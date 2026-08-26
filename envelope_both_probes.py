"""
Envelope Phi_det at BOTH ceilings, six seeds per tube width, on the paper's own
convention (deficit_floor_law.integrate, t0=1e-6, cap_mode='min').

Answers two questions the single-seed run could not:
  (1) how much of the measured single-pass floor the envelope accounts for, read at the
      forward-process ceiling versus at the sampler-law ceiling of eq. (3);
  (2) whether the comparison survives at every tube width tested.

It does not survive at the widest. That is reported, not smoothed.

Run:  python envelope_both_probes.py     ->  envelope_both_probes.jsonl
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import kbar_sampler_law as K

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "envelope_both_probes.jsonl")
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
SEEDS = range(6)


def integrate(t0, kbar, sigma, KK=600000, tstart=8.0):
    """deficit_floor_law.integrate, cap_mode='min'."""
    sig2 = sigma * sigma
    ks = lambda t: (lambda a2, s: s / (a2 * sig2 + s * s))(np.exp(-t), np.sqrt(1 - np.exp(-t)))
    ts = np.geomspace(tstart, t0, KK + 1); V = 1.0
    for i in range(KK):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        V = V + ((1 - 2 * min(ks(t), kbar) / s) * V + 1) * dt
        if V < 0:
            V = 1e-12
    return V


if __name__ == "__main__":
    t0_ = time.time(); recs = []
    print(f"{'sigma':>6} {'probe':>8} {'kbar':>16} {'Phi/sig^2':>16} "
          f"{'frac of measured':>18} {'viol':>6}")
    for sig in SIGS:
        s2 = sig * sig
        kp, ka, ph_p, ph_a, vv = [], [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(500 + sd)
            net = K.train_on(K.ring(K.N, sig, rng), 500 + sd)
            a = K.kbar_pt(net, sig)[0]
            b, _, _, v = K.kbar_sampler(net, seed=11 + sd)
            kp.append(a); ka.append(b); vv.append(v / s2)
            ph_p.append(integrate(1e-6, a, sig) / s2)
            ph_a.append(integrate(1e-6, b, sig) / s2)
        for tag, kk, pp in (("p_t", kp, ph_p), ("sampler", ka, ph_a)):
            fr = [p / v for p, v in zip(pp, vv)]
            viol = sum(1 for p, v in zip(pp, vv) if p > v)
            print(f"{sig:6.2f} {tag:>8} {np.mean(kk):6.2f}[{min(kk):.2f},{max(kk):.2f}] "
                  f"{np.mean(pp):7.2f}[{min(pp):.2f},{max(pp):.2f}] "
                  f"{100*np.mean(fr):8.0f}% [{100*min(fr):.0f},{100*max(fr):.0f}] "
                  f"{viol:>4}/6")
            recs.append(dict(key=f"EBP_{sig}_{tag}", sigma=sig, probe=tag,
                             kbar_mean=round(float(np.mean(kk)), 4),
                             kbar_range=[round(min(kk), 3), round(max(kk), 3)],
                             phi_mean=round(float(np.mean(pp)), 4),
                             frac_mean=round(float(np.mean(fr)), 4),
                             frac_range=[round(min(fr), 4), round(max(fr), 4)],
                             violations=viol, n_seeds=len(list(SEEDS))))
        print()
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}  ({time.time()-t0_:.0f}s)")
