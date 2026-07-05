"""Diagnose the Arm-A theory gap: my pre-registered v*_det=4.43 sigma^2 integrated the
deficit-floor ODE to t0=1e-6, but Cambridge's annealed Arm A actually stops at t0=1e-4
by its final generation. Integrating further than the method runs under-estimates the
residual floor. Recompute the deterministic recursion fixed point at the METHOD'S real
truncation t0, with a small kbar/t0 sensitivity grid. Data to match: 9.84 sigma^2."""
import numpy as np

SIGMA = 0.05; SIG2 = SIGMA ** 2; LAM = 0.5

def Phi_det(w, kbar, t0, K=80000, tstart=8.0):
    def kstar(t):
        a2 = np.exp(-t); s = np.sqrt(1 - a2); return s / (a2 * w + s * s)
    ts = np.geomspace(tstart, t0, K + 1); V = 1.0
    for i in range(K):
        t = ts[i]; dt = ts[i] - ts[i + 1]; s = np.sqrt(1 - np.exp(-t))
        k = min(kstar(t), kbar); V = V + ((1 - 2 * k / s) * V + 1) * dt
        if V < 0: V = 1e-12
    return V

def fixed_point(kbar, t0):
    v = 0.01
    for _ in range(60):
        vn = Phi_det(LAM * SIG2 + (1 - LAM) * v, kbar, t0)
        if abs(vn - v) < 1e-6: v = vn; break
        v = vn
    return v

print(f"sigma^2={SIG2}   DATA (Arm A g19) = 0.02460 = 9.84 sigma^2\n")
print(f"{'t0':>8} {'kbar':>6} {'v*':>10} {'v*/sig2':>9}")
for t0 in [1e-6, 1e-4, 3e-4, 1e-3]:
    for kbar in [3.5, 3.9, 4.5]:
        v = fixed_point(kbar, t0)
        flag = "  <-- Arm A t0, measured kbar" if (t0 == 1e-4 and kbar == 3.9) else ""
        print(f"{t0:>8.0e} {kbar:>6.1f} {v:>10.5f} {v/SIG2:>8.2f}x{flag}")
