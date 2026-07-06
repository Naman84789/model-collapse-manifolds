"""Fast smoke test before committing to the full cifar_recursion.py run.
Checks: CIFAR-10 downloads/loads, UNet forward/backward on GPU, sampler runs,
matcher runs without shape errors. Uses tiny STEPS/G so it finishes in ~1-2 min.
"""
import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] importing cifar_recursion module pieces...", flush=True)

# import the module but prevent its top-level training loop from running by
# temporarily faking SEEDS to empty via monkeypatch after partial import isn't
# possible cleanly, so instead we re-implement the smoke path using the same
# building blocks via exec of just the needed defs. Simplest robust approach:
# import as __main__ guard isn't present in the script, so we patch sys.argv
# trick won't help either. Instead, monkey-patch by importing with SEEDS=[]
# using a copy trick: read source, exec with SEEDS overridden.

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cifar_recursion.py")).read()
src = src.replace("SEEDS = [0, 1, 2]", "SEEDS = []")          # skip the real run
src = src.replace("STEPS, BATCH, G, LAM, NSAMP = 8.0, 6000, 128, 8, 0.5, 6000",
                  "STEPS, BATCH, G, LAM, NSAMP = 8.0, 6000, 128, 8, 0.5, 6000")
ns = {"__name__": "smoke", "__file__": os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cifar_recursion.py")}
exec(compile(src, "cifar_recursion.py", "exec"), ns)
print(f"[{time.time()-t0:.1f}s] module loaded, data ready. DEV={ns['DEV']}", flush=True)
print(f"  TRAIN shape={ns['TRAIN'].shape}  TEST shape={ns['TEST'].shape}  TRUE baseline={ns['TRUE']:.4f}",
      flush=True)

# ---- tiny training step (50 steps instead of 6000) ----
import torch
print(f"[{time.time()-t0:.1f}s] training UNet for 50 steps (smoke)...", flush=True)
tick = time.time()
net = ns["UNet"]().to(ns["DEV"])
opt = torch.optim.Adam(net.parameters(), 2e-4)
data = torch.tensor(ns["TRAIN"][:2000].reshape(-1, 3, 32, 32))
scaler = torch.amp.GradScaler(ns["DEV"]) if ns["DEV"] == "cuda" else None
for step in range(50):
    idx = torch.randint(0, len(data), (64,))
    x0 = data[idx].to(ns["DEV"])
    t = torch.rand(64, device=ns["DEV"]) * (8.0 - 1e-3) + 1e-3
    a = torch.exp(-t / 2)[:, None, None, None]
    s = torch.sqrt(1 - torch.exp(-t))[:, None, None, None]
    z = torch.randn_like(x0); xt = a * x0 + s * z
    if scaler:
        with torch.amp.autocast(ns["DEV"]):
            loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        loss = ((net(xt, t) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
dt = time.time() - tick
print(f"[{time.time()-t0:.1f}s] 50 steps in {dt:.2f}s ({50/dt:.1f} it/s), loss={loss.item():.4f}",
      flush=True)
print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB" if ns["DEV"]=="cuda" else "  CPU mode", flush=True)

# ---- tiny sampling step ----
print(f"[{time.time()-t0:.1f}s] sampling 32 images (smoke, 40-step grid)...", flush=True)
tick = time.time()
samples = ns["std_sample"](net, 32, 0.02, seed=0, k=40, chunk=32)
print(f"[{time.time()-t0:.1f}s] sampled in {time.time()-tick:.2f}s, shape={samples.shape}", flush=True)

# ---- matcher smoke test ----
print(f"[{time.time()-t0:.1f}s] testing matcher (bidirectional, D=3072)...", flush=True)
tick = time.time()
ref = ns["TRAIN"][:500]
matched = ns["axmatch_bidir"](samples, ref, k=160, kdim=24, jax=128)
print(f"[{time.time()-t0:.1f}s] matcher ran in {time.time()-tick:.2f}s, shape={matched.shape}, "
      f"no NaN={not __import__('numpy').isnan(matched).any()}", flush=True)

# ---- offman metric + grid save ----
om = ns["offman"](samples, ns["TESTREF"])
om_matched = ns["offman"](matched, ns["TESTREF"])
print(f"[{time.time()-t0:.1f}s] offman(unmatched, undertrained)={om:.4f}  "
      f"offman(matched)={om_matched:.4f}  true_baseline={ns['TRUE']:.4f}", flush=True)

ns["save_grid"](samples, os.path.join(ns["BASE"], "smoke_grid.png"), n=32)
print(f"[{time.time()-t0:.1f}s] SMOKE TEST PASSED -- all pieces work end-to-end", flush=True)
