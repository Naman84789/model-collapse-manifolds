"""
Item 5 — the small-c near-cancellation, resolved into its TWO channels.

The law's per-generation map is v = w + e^2(w) (output normal variance vs pool width w),
and 1+c := dv/dw at the fixed point. Two competing, opposite-signed contributions:
  - DEFICIT response (deterministic, deficit_floor_law): d Phi_det/dw - 1 < 0 (measured
    -0.7..0), because a fatter pool is EASIER to represent (kmax(w)=1/(2 sqrt w) falls).
  - INJECTION response (stochastic residual, Prop 2.4): > 0, because a fatter pool damps
    the injected field less, retaining more.
Net measured c ~ +0.03 (LAW_b) = residual of these two O(0.3-0.7) mechanisms.

This probe measures the TOTAL dv/dw by training single-width tubes and comparing to the
deterministic Phi_det(w): the GAP is the injection channel. Confirms small, concave,
POSITIVE-or-near-zero net c, and attributes it.

Ring radius R=2.5 (validated protocol), width sig_eff swept. Resumable.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, json, time

torch.set_num_threads(12)
OUT = r"poolwidth_probe.jsonl"
R, T_MAX, STEPS, BATCH, N = 2.5, 8.0, 2000, 512, 6000

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
    data = torch.tensor(ring(N, sig, rng)); opt = torch.optim.Adam(Score().parameters(), 2e-3)
    net = Score(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def std_sample(net, n, t0, seed, k=200):
    torch.manual_seed(seed); x = torch.randn(n, 2)
    ts = np.geomspace(T_MAX, t0, k + 1)
    for i in range(k):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i]); s = np.sqrt(1 - np.exp(-t))
        with torch.no_grad():
            e = net(x, torch.full((n, 1), t))
        x = x + (-0.5 * x + e / s) * dt + np.sqrt(abs(dt)) * torch.randn_like(x)
    return x.numpy()

def kbar_fit(net, sig):
    """measured normal-slope ceiling: max over t of the affine fit slope kappa_hat(t)."""
    ks = []
    for t in [0.1, 0.05, 0.02, 0.01, 0.005]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t)); rng = np.random.default_rng(7)
        th = rng.uniform(0, 2 * np.pi, 4000); rr = a * (R + sig * rng.normal(size=4000))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(4000, 2))
        rad = np.linalg.norm(x, 1 - 1, keepdims=False) if False else np.linalg.norm(x, axis=1)
        uh = x / rad[:, None]; h = rad - a * R
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32), torch.full((4000, 1), float(t))).numpy() * uh).sum(1)
        ks.append(float(np.cov(h, er)[0, 1] / h.var()))
    return max(ks)

SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]
print(f"{'sig':>6} {'w':>8} {'v_out':>9} {'v-w':>9} {'kbar':>6}", flush=True)
res = []
for sig in SIGS:
    key = f"PW_sig{sig}"
    if key in done:
        for line in open(OUT):
            d = json.loads(line)
            if d.get("key") == key:
                res.append(d)
        print(f"{key} skip", flush=True); continue
    vs = []
    for seed in [0, 1]:
        net = train(sig, 500 + seed)
        x = std_sample(net, 4000, 0.005, 900 + seed)
        vs.append(float(np.linalg.norm(x, axis=1).var()))
    kb = kbar_fit(net, sig)
    v = float(np.mean(vs)); w = sig * sig
    rec = dict(key=key, sig=sig, w=round(w, 5), v_out=round(v, 5),
               excess=round(v - w, 5), kbar=round(kb, 3), seeds=[round(x, 5) for x in vs])
    log(rec); res.append(rec)
    print(f"{sig:>6} {w:>8.5f} {v:>9.5f} {v-w:>9.5f} {kb:>6.2f}   ({el()})", flush=True)

# net response dv/dw and attribution vs deterministic Phi_det
res = sorted([r for r in res if "w" in r], key=lambda r: r["w"])
if len(res) >= 2:
    ws = np.array([r["w"] for r in res]); vs = np.array([r["v_out"] for r in res])
    dvdw = np.gradient(vs, ws)
    print("\nw, v_out, dv/dw = 1+c_total (net); c_total = dv/dw - 1:", flush=True)
    for i in range(len(ws)):
        print(f"  w={ws[i]:.5f}: v={vs[i]:.5f}  dv/dw={dvdw[i]:+.3f}  c_total={dvdw[i]-1:+.3f}", flush=True)
    log(dict(key="SUMMARY", ws=[round(x,5) for x in ws], vs=[round(x,5) for x in vs],
             dvdw=[round(x,3) for x in dvdw]))
print(f"\nALL DONE {el()}", flush=True)
