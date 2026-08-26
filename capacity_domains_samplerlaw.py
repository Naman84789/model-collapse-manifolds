"""
capacity_domains.py re-measured under the SAMPLER's own law.

The three extra geometries carry the claim that the degradation law is about
representing the local normal score, not about rings. Their ceilings are read off
the FORWARD process; eq. (3) asks for the slope under the sampler's own ensemble.
The claim is relative (comparable percentage drop across sqrt(w)), so it survives a
constant probe ratio -- but the ratio is NOT constant: it rises with tube width on
ring tubes (0.823 -> 0.859) and splits by training protocol on the interventional
arms (0.754 -> 0.853). Whether it also varies with codimension is unmeasured, and
RING10 puts nine normal directions on the network at once.

Measures both probes on the same networks, per geometry, and reports the relative
drop under each.

Run:  python capacity_domains_samplerlaw.py  ->  capacity_domains_samplerlaw.jsonl
Resumable: appends per (geometry, width) and skips completed ones.
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "capacity_domains_samplerlaw.jsonl")

R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
L = 2 * np.pi * R
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
TS_FIT = [0.1, 0.05, 0.02, 0.01, 0.005]
T0, KGRID = 1e-4, 200
TS = np.geomspace(T_MAX, T0, KGRID + 1)
SEED = 500


def temb(t, dim=32):
    f = torch.exp(torch.linspace(0, 5, dim // 2)).to(t)
    return torch.cat([torch.sin(t * f), torch.cos(t * f)], 1)


class Score(nn.Module):
    def __init__(self, D=2, h=256, te=32):
        super().__init__(); self.te = te
        self.net = nn.Sequential(
            nn.Linear(D + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def forward(self, x, t):
        return self.net(torch.cat([x, temb(t, self.te)], 1))


def data_seg(n, sig, rng):
    x = rng.uniform(-L / 2, L / 2, n); y = sig * rng.normal(size=n)
    return np.stack([x, y], 1).astype("float32")


def data_sphere(n, sig, rng):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * (R + sig * rng.normal(size=(n, 1)))).astype("float32")


def data_ring10(n, sig, rng):
    # verbatim from capacity_domains.py: radial spread IN-PLANE plus 8 out-of-plane
    # directions (9 normal directions). Adding isotropic noise to all 10 coords
    # instead would smear along the manifold, not just normal to it.
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    out = np.zeros((n, 10), dtype="float32")
    out[:, 0] = rr * np.cos(th); out[:, 1] = rr * np.sin(th)
    out[:, 2:] = sig * rng.normal(size=(n, 8))
    return out


GEOMS = {"SEG": (data_seg, 2), "SPHERE": (data_sphere, 3), "RING10": (data_ring10, 10)}


def train(geom, sig, seed):
    maker, D = GEOMS[geom]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(maker(N, sig, rng))
    net = Score(D); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, D); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def _slope_from(geom, x, e, mask=None):
    """same normal coordinate convention as capacity_domains.slope_at"""
    if geom == "SEG":
        m = np.ones(len(x), bool) if mask is None else mask
        h = x[m, 1] - x[m, 1].mean()
        return float(np.cov(h, e[m, 1])[0, 1] / max(h.var(), 1e-12))
    if geom == "SPHERE":
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        h = rad - rad.mean(); er = (e * uh).sum(1)
        return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))
    ks = []
    for j in range(2, 10):
        h = x[:, j] - x[:, j].mean()
        ks.append(np.cov(h, e[:, j])[0, 1] / max(h.var(), 1e-12))
    return float(np.mean(ks))


def kbar_pt(net, geom, sig, n=4000, seed=7):
    maker, D = GEOMS[geom]
    ks = []
    for t in TS_FIT:
        rng = np.random.default_rng(seed)
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        x0 = maker(n, sig, rng)
        x = a * x0 + s * rng.normal(size=(n, D)).astype("float32")
        with torch.no_grad():
            e = net(torch.tensor(x), torch.full((n, 1), float(t))).numpy()
        mask = np.abs(a * x0[:, 0]) < L / 6 if geom == "SEG" else None
        ks.append(_slope_from(geom, x, e, mask))
    return max(ks)


def kbar_sampler(net, geom, n=4000, seed=11):
    """Run the reverse sampler in D dims; read the slope off the live ensemble."""
    maker, D = GEOMS[geom]
    torch.manual_seed(seed); x = torch.randn(n, D)
    want = {min(range(KGRID), key=lambda i: abs(TS[i] - p)) for p in TS_FIT}
    ks = []
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        if i in want:
            xn = x.numpy()
            mask = np.abs(xn[:, 0]) < L / 6 if geom == "SEG" else None
            ks.append(_slope_from(geom, xn, e.numpy(), mask))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return max(ks)


if __name__ == "__main__":
    t_start = time.time()
    done, recs = set(), []
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); recs.append(r); done.add(r["key"])
            except Exception:
                pass
    print(f"{'geom':>7} {'sig':>5} {'kb_pt':>7} {'kb_samp':>8} {'ratio':>6}")
    for geom in GEOMS:
        for sig in SIGS:
            key = f"CDS_{geom}_{sig}"
            if key in done:
                r = next(x for x in recs if x["key"] == key)
                print(f"{geom:>7} {sig:5.2f} {r['kbar_pt']:7.3f} "
                      f"{r['kbar_sampler']:8.3f} {r['ratio']:6.3f}  (cached)")
                continue
            net = train(geom, sig, SEED)
            a = kbar_pt(net, geom, sig)
            b = kbar_sampler(net, geom)
            rec = dict(key=key, geom=geom, sig=sig, w=round(sig * sig, 6),
                       kbar_pt=round(a, 4), kbar_sampler=round(b, 4),
                       ratio=round(b / a, 4))
            recs.append(rec)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{geom:>7} {sig:5.2f} {a:7.3f} {b:8.3f} {b/a:6.3f}")
    print()
    print("relative drop across sqrt(w) = 0.03 -> 0.13, under each probe")
    print(f"{'geom':>7} {'drop_pt':>9} {'drop_samp':>11} {'ratio lo':>9} {'ratio hi':>9}")
    for geom in GEOMS:
        rs = sorted([r for r in recs if r["geom"] == geom], key=lambda r: r["sig"])
        if len(rs) < 2:
            continue
        dp = 1 - rs[-1]["kbar_pt"] / rs[0]["kbar_pt"]
        ds = 1 - rs[-1]["kbar_sampler"] / rs[0]["kbar_sampler"]
        rr = [r["ratio"] for r in rs]
        print(f"{geom:>7} {100*dp:8.1f}% {100*ds:10.1f}% {min(rr):9.3f} {max(rr):9.3f}")
    print(f"\nwrote {OUT}  ({time.time()-t_start:.0f}s)")
