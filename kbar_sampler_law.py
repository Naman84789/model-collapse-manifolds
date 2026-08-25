"""
kbar measured under the SAMPLER's own law, not under p_t.

Theorem 2 is stated on
    khat(t) := Cov(X_t, eps_hat(X_t,t)) / Var(X_t)
with X_t the sampler's own ensemble at noise level t. The published probe
(gap_review_torch.kbar_fit, ceiling_origin.profile_pt) instead draws the probe
points from the FORWARD process, i.e. from p_t. For an affine estimator the two
coincide; for a nonlinear one they do not, because the regression slope depends
on the law it is taken under.

This script measures both on the same trained networks and reports the gap, so
the floor quoted in the paper can be recomputed against the quantity the theorem
actually constrains.

Run:  python kbar_sampler_law.py
Writes kbar_sampler_law.jsonl
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(8)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "kbar_sampler_law.jsonl")

# geometry / training identical to gap_review_torch.py
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
T0, KGRID = 1e-4, 200
TS = np.geomspace(T_MAX, T0, KGRID + 1)
PROBE_TS = [0.1, 0.05, 0.02, 0.01, 0.005]      # the published probe's five times
BAND = (0.004, 0.12)                            # band for the max-over-band variant


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


def ring(n, sig, rng):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")


def train_on(data_np, seed):
    torch.manual_seed(seed)
    data = torch.tensor(data_np)
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, len(data), (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def _slope(x_np, er):
    """Regression slope of the normal eps-component on the normal coordinate."""
    rad = np.linalg.norm(x_np, axis=1)
    h = rad - rad.mean()
    return float(np.cov(h, er)[0, 1] / max(h.var(), 1e-12))


def kbar_pt(net, sig_for_probe):
    """PUBLISHED probe: points drawn from the forward process (approximately p_t)."""
    ks = []
    for t in PROBE_TS:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        rng = np.random.default_rng(7)
        th = rng.uniform(0, 2 * np.pi, 4000)
        rr = a * (R + sig_for_probe * rng.normal(size=4000))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(4000, 2))
        rad = np.linalg.norm(x, axis=1); uh = x / rad[:, None]
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((4000, 1), float(t))).numpy() * uh).sum(1)
        ks.append(_slope(x, er))
    return max(ks), ks


def kbar_sampler(net, n=4000, seed=11):
    """THEOREM's quantity: slope on the sampler's own ensemble, measured in-flight.

    Runs the same reverse-SDE sampler the paper uses and reads the regression
    slope off the live ensemble just before each step. Returns the max at the
    five published probe times, the max over the whole band, and the final
    ensemble variance for context.
    """
    torch.manual_seed(seed)
    x = torch.randn(n, 2)
    idx_for = {min(range(KGRID), key=lambda i: abs(TS[i] - pt)): pt for pt in PROBE_TS}
    at_probe, in_band, traj = [], [], []
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        xn = x.numpy(); rad = np.linalg.norm(xn, axis=1)
        uh = xn / rad[:, None]
        er = (e.numpy() * uh).sum(1)
        k = _slope(xn, er)
        traj.append((t, k))
        if BAND[0] <= t <= BAND[1]:
            in_band.append(k)
        if i in idx_for:
            at_probe.append(k)
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    v_end = float(np.linalg.norm(x.numpy(), axis=1).var())
    return max(at_probe), max(in_band) if in_band else float("nan"), at_probe, v_end


if __name__ == "__main__":
    t_start = time.time()
    SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
    print("kbar under p_t (published probe) vs under the sampler's own law")
    print(f"{'sigma':>6} {'kbar_pt':>9} {'kbar_samp':>10} {'ratio':>7} "
          f"{'floor_pt':>10} {'floor_samp':>11} {'V_end/s^2':>10}")
    recs = []
    for sig in SIGS:
        rng = np.random.default_rng(500)
        net = train_on(ring(N, sig, rng), 500)
        kpt, kpt_all = kbar_pt(net, sig)
        ksamp, ksamp_band, ksamp_all, v_end = kbar_sampler(net)
        f_pt = 1.0 / (8 * kpt ** 2) / sig ** 2
        f_sa = 1.0 / (8 * ksamp ** 2) / sig ** 2
        print(f"{sig:6.2f} {kpt:9.3f} {ksamp:10.3f} {ksamp/kpt:7.3f} "
              f"{f_pt:10.3f} {f_sa:11.3f} {v_end/sig**2:10.3f}")
        rec = dict(key=f"KSL_{sig}", sigma=sig, kbar_pt=kpt, kbar_sampler=ksamp,
                   kbar_sampler_band=ksamp_band, ratio=ksamp / kpt,
                   floor_pt_s2=f_pt, floor_sampler_s2=f_sa, v_end_s2=v_end / sig ** 2,
                   kbar_pt_profile=kpt_all, kbar_sampler_profile=ksamp_all)
        recs.append(rec)
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}   ({time.time()-t_start:.0f}s)")
    print("floor_* are 1/(8 kbar^2) in sigma^2 units; V_end is the measured sampler variance.")
    print("A floor ABOVE V_end would falsify the theorem at that kbar.")
