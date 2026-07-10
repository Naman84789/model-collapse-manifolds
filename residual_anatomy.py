"""
RESIDUAL ANATOMY -- attack on open problem (2): the 1.5-2.0x recursion residual.

The self-consistent closure predicts the recursion fixed point from the ENVELOPE
profile khat(t) = min(kappa(t;w), kbar(w)).  Three channels it ignores, none ever
measured where they matter (inside the deficit band, at the fixed-point pool width,
under the sampler's own ensemble):

  C1 PROFILE SHAPE : the real khat(t) may sag below its own ceiling inside the band
                     and under-track (0.95-0.97) outside it.
  C2 ENSEMBLE LAG  : khat is defined by regression under p_t; the sampler's ensemble
                     is NOT p_t once it has drifted, so the slope it actually
                     experiences differs (this is the orthogonality shield's
                     second-order term, made visible).
  C3 NONLINEAR KICK: the orthogonal residual delta injects variance (the ~30% single-
                     pass remainder on the clean tube).

Design, per training width sig_p in {0.05, 0.07, 0.095, 0.11, 0.122} x 2 net seeds:
  v_env  : discrete variance recursion on the sampler's own 200-pt grid with the
           envelope profile (ceiling = this net's own measured kbar)  [closure model]
  v_prof : same recursion with the finely MEASURED p_t profile khat(t) (all 200 grid
           times, regression on 4000 forward-diffused points each)     [adds C1]
  v_ens  : same recursion with the slope the sampler ACTUALLY experienced -- measured
           by regressing the net's radial output on the ensemble's own radial
           deviations at every step of a real sampling run              [adds C2]
  v_full : the real sampler's output radial variance (3 sampler seeds) [adds C3]
Channel sizes: C1 = v_prof - v_env, C2 = v_ens - v_prof, C3 = v_full - v_ens.

RECURSION CHECK: iterate the MEASURED map  v <- v_full(w(v)),  w(v) = 0.5 sig^2 +
0.5 v  (lambda=1/2, interpolated over the five widths).  PREDICTION if A0 holds and
the channels are the whole story: the fixed point lands in the measured ablation
band 10.8-13.2 sigma^2 (sigma=0.05), closing the 1.5-2.0x factor into measured,
attributed channels.  t0 = 1e-4 throughout (the ablation's operating point).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(6)
OUT = r"C:\Users\naman\Downloads\metric-audit\residual_anatomy.jsonl"
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000
SIG0 = 0.05                       # the recursion's data width (sigma^2 = 0.0025)
T0, KGRID = 1e-4, 200
SIGS = [0.05, 0.07, 0.095, 0.11, 0.122, 0.135]
NET_SEEDS = [500, 501]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            pass

def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")

t0_ = time.time()
def el():
    return f"{(time.time()-t0_)/60:.1f} min"

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

def train(sig, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = torch.tensor(ring(N, sig, rng))
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

TS = np.geomspace(T_MAX, T0, KGRID + 1)

def radial_fit(net, x, t):
    """affine fit of the net's radial output against radial deviation of x."""
    rad = np.linalg.norm(x, axis=1); uh = x / np.maximum(rad[:, None], 1e-9)
    with torch.no_grad():
        er = (net(torch.tensor(x, dtype=torch.float32),
                  torch.full((len(x), 1), float(t))).numpy() * uh).sum(1)
    h = rad - rad.mean()
    vh = h.var()
    k = float(np.cov(h, er)[0, 1] / max(vh, 1e-12))
    resid = er - k * h - er.mean()
    return k, float((resid ** 2).mean()), float(vh)

def profile_pt(net, sig, n=4000, seed=7):
    """measured khat(t) + residual risk r2(t) under p_t, on the sampler grid."""
    rng = np.random.default_rng(seed)
    ks, r2s = [], []
    for t in TS[:-1]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t))
        th = rng.uniform(0, 2 * np.pi, n)
        rr = a * (R + sig * rng.normal(size=n))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(n, 2))
        k, r2, _ = radial_fit(net, x, t)
        ks.append(k); r2s.append(r2)
    return np.array(ks), np.array(r2s)

def sample_instrumented(net, n, seed):
    """real sampler; also record the slope the ensemble actually experienced."""
    torch.manual_seed(seed); x = torch.randn(n, 2)
    k_ens = []
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        k, _, _ = radial_fit(net, x.numpy(), t)
        k_ens.append(k)
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    v = float(np.linalg.norm(x.numpy(), axis=1).var())
    return v, np.array(k_ens)

def v_discrete(kfun_vals, sig):
    """variance of the sampler's own discrete affine recursion, grid-exact.
    Init = radial variance of the actual N(0,I_2) start (Rayleigh), 2 - pi/2;
    the choice is contracted away by ~e^-7 before the band (checked)."""
    V = 2.0 - np.pi / 2.0
    for i in range(KGRID):
        t = float(TS[i]); dt = float(TS[i + 1] - TS[i]); s = np.sqrt(1 - np.exp(-t))
        f = 1.0 + (kfun_vals[i] / s - 0.5) * dt          # dt < 0
        V = f * f * V + abs(dt)
    return float(V)

def kappa_true(t, sig):
    a2 = np.exp(-t); s2 = 1 - np.exp(-t)
    return np.sqrt(s2) / (a2 * sig * sig + s2)

print("width  kbar   v_env   v_prof  v_ens   v_full   (units of sigma0^2=0.0025)", flush=True)
recs = []
for sig in SIGS:
    for ns in NET_SEEDS:
        key = f"RA_sig{sig}_n{ns}"
        if key in done:
            for line in open(OUT):
                d = json.loads(line)
                if d.get("key") == key:
                    recs.append(d)
            print(f"{key} skip", flush=True); continue
        net = train(sig, ns)
        ks_pt, r2_pt = profile_pt(net, sig)
        kbar = float(ks_pt.max())
        band = kappa_true(TS[:-1], sig) > kbar
        sag = float((kbar - ks_pt[band]).max()) if band.any() else 0.0
        env = np.minimum(kappa_true(TS[:-1], sig), kbar)
        v_env = v_discrete(env, sig)
        v_prof = v_discrete(ks_pt, sig)
        vfl, kel = [], []
        for ss in (0, 1, 2):
            vf, k_ens = sample_instrumented(net, 4000, 900 + 10 * ns + ss)
            vfl.append(vf); kel.append(k_ens)
        k_ens = np.mean(kel, axis=0)
        v_ens = v_discrete(k_ens, sig)
        v_full = float(np.mean(vfl))
        rec = dict(key=key, sig=sig, w=round(sig * sig, 6), net_seed=ns,
                   kbar=round(kbar, 3), sag_in_band=round(sag, 3),
                   v_env=round(v_env, 6), v_prof=round(v_prof, 6),
                   v_ens=round(v_ens, 6), v_full=round(v_full, 6),
                   v_full_seeds=[round(v, 6) for v in vfl],
                   khat_pt=[round(float(k), 4) for k in ks_pt],
                   khat_ens=[round(float(k), 4) for k in k_ens],
                   r2_pt=[round(float(r), 5) for r in r2_pt],
                   ts=[round(float(t), 6) for t in TS[:-1]])
        log(rec); recs.append(rec)
        u = SIG0 ** 2
        print(f"{sig:>5} {kbar:>5.2f} {v_env/u:>7.2f} {v_prof/u:>7.2f} "
              f"{v_ens/u:>7.2f} {v_full/u:>7.2f}   sag={sag:.2f}  ({el()})", flush=True)

# ---- channel table + measured-map recursion --------------------------------
by_sig = {}
for r in recs:
    by_sig.setdefault(r["sig"], []).append(r)
u = SIG0 ** 2
ws, vfull_m = [], []
print("\nCHANNELS (mean over net seeds, units sigma0^2):", flush=True)
print("width    C1 profile   C2 ensemble   C3 nonlinear   total gap", flush=True)
for sig in SIGS:
    rs = by_sig.get(sig, [])
    if not rs:
        continue
    ve = np.mean([r["v_env"] for r in rs]); vp = np.mean([r["v_prof"] for r in rs])
    vn = np.mean([r["v_ens"] for r in rs]); vf = np.mean([r["v_full"] for r in rs])
    print(f"{sig:>5}   {(vp-ve)/u:>+9.2f}   {(vn-vp)/u:>+9.2f}   {(vf-vn)/u:>+9.2f}"
          f"   {(vf-ve)/u:>+9.2f}", flush=True)
    ws.append(sig * sig); vfull_m.append(vf)

ws = np.array(ws); vfull_m = np.array(vfull_m)
lam = 0.5
v = SIG0 ** 2
for it in range(400):
    w = lam * SIG0 ** 2 + (1 - lam) * v
    v = float(np.interp(w, ws, vfull_m))
fp = v / u
print(f"\nMEASURED-MAP RECURSION FIXED POINT (lambda=1/2, t0=1e-4): {fp:.2f} sigma^2", flush=True)
print("ablation band to hit: 10.8-13.2 sigma^2", flush=True)
log(dict(key="SUMMARY", ws=[round(float(x), 6) for x in ws],
         v_full=[round(float(x), 6) for x in vfull_m], fixed_point_sig2=round(fp, 3)))
print(f"ALL DONE {el()}", flush=True)
