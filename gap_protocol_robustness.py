"""
Does the capacity degradation kbar(w) replicate OUTSIDE the protocol it was measured
on?  (The referee attack: "is there any independent setting where the same law
appears?")  Three variations, none used in the original probe:

  P1  half-width network (h=128 instead of 256), same ring R=2.5
  P2  different geometry (ring R=3.5 instead of 2.5), same network h=256
  P3  a third training seed (777) on the baseline protocol (error-bar densification;
      the probe logged seed-501 nets, the review added seed-500)

The claim under test is the PHENOMENON (kbar falls roughly linearly in sqrt(w), with
comparable slope), not the constants (the level is protocol-specific).
Run with the SYSTEM python (torch lives there, not in the venv).
"""
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(12)
T_MAX, STEPS, BATCH, N = 8.0, 2000, 512, 6000
SIGS = [0.03, 0.05, 0.07, 0.10, 0.13]

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

def ring(n, sig, rng, R):
    th = rng.uniform(0, 2 * np.pi, n); rr = R + sig * rng.normal(size=n)
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype("float32")

def train_on(data_np, seed, h):
    torch.manual_seed(seed)
    data = torch.tensor(data_np)
    net = Score(h=h); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, len(data), (BATCH,)); x0 = data[idx]
        t = torch.rand(BATCH, 1) * (T_MAX - 1e-3) + 1e-3
        a = torch.exp(-t / 2); s = torch.sqrt(1 - torch.exp(-t))
        z = torch.randn(BATCH, 2); xt = a * x0 + s * z
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def kbar_fit(net, sig, R):
    ks = []
    for t in [0.1, 0.05, 0.02, 0.01, 0.005]:
        a = np.exp(-t / 2); s = np.sqrt(1 - np.exp(-t)); rng = np.random.default_rng(7)
        th = rng.uniform(0, 2 * np.pi, 4000)
        rr = a * (R + sig * rng.normal(size=4000))
        x = np.stack([rr * np.cos(th), rr * np.sin(th)], 1) + s * rng.normal(size=(4000, 2))
        rad = np.linalg.norm(x, axis=1)
        uh = x / rad[:, None]; h = rad - a * R
        with torch.no_grad():
            er = (net(torch.tensor(x, dtype=torch.float32),
                      torch.full((4000, 1), float(t))).numpy() * uh).sum(1)
        ks.append(float(np.cov(h, er)[0, 1] / h.var()))
    return max(ks)

t0_ = time.time()
CONFIGS = [("P1 h=128 R=2.5", 128, 2.5, 500),
           ("P2 h=256 R=3.5", 256, 3.5, 500),
           ("P3 h=256 R=2.5 seed777", 256, 2.5, 777)]
print("baseline laws for comparison: seed501 4.434 -11.641 sqrt(w); seed500 4.056 -9.688 sqrt(w)\n")
for name, h, R, seed in CONFIGS:
    kbs = []
    for sig in SIGS:
        rng = np.random.default_rng(seed)
        net = train_on(ring(N, sig, rng, R), seed, h)
        kb = kbar_fit(net, sig, R)
        kbs.append(kb)
        print(f"  {name}: sig={sig:.2f} kbar={kb:.3f}   [{(time.time()-t0_)/60:.1f} min]", flush=True)
    sl, ic = np.polyfit(np.sqrt(np.array(SIGS) ** 2), kbs, 1)  # x = sig
    resid = np.array(kbs) - (ic + sl * np.array(SIGS))
    print(f"  {name}: law kbar = {ic:.3f} {sl:+.3f} sqrt(w)   max|resid| = {np.abs(resid).max():.3f}"
          f"   degradation over range: {kbs[0]:.2f} -> {kbs[-1]:.2f}\n", flush=True)
print("DONE", flush=True)
