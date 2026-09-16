"""Generate the 3D homogeneous FDTD reference without storing the full fine grid history.

The paper configuration uses dx=2.5e-3, nt=450, c=0.5, and three sources. Only the
151 x 101 x 101 x 101 evaluation array is stored.
"""
from pathlib import Path
import time
import numpy as np
from numba import njit, prange

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

c, L, T = 0.5, 1.0, 1.0
dx = 0.0025
nt = 450
dt = T / nt
nx = int(round(L / dx)) + 1
cfl = c * dt * np.sqrt(3.0) / dx
if cfl > 1.0:
    raise RuntimeError(f"CFL condition violated: {cfl:.6f}")

x = np.linspace(0.0, L, nx)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
g1 = np.exp(-((X-0.1)**2 + (Y-0.6)**2 + (Z-0.1)**2)/(2*0.2**2))
g2 = np.exp(-((X-0.7)**2 + (Y-0.2)**2 + (Z-0.4)**2)/(2*0.2**2))
g3 = np.exp(-((X-0.4)**2 + (Y-0.8)**2 + (Z-0.9)**2)/(2*0.3**2))
del X, Y, Z

u_prev = np.zeros((nx,nx,nx), dtype=np.float64)
u_now = np.zeros_like(u_prev)
u_next = np.zeros_like(u_prev)
coef = (c*dt/dx)**2

@njit(parallel=True)
def step(u_prev, u_now, u_next, g1, g2, g3, coef, dt, t):
    n = u_now.shape[0]
    tg = np.exp(-((t-0.2)**2)/(2*0.2**2))
    for i in prange(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1):
                lap = (
                    u_now[i+1,j,k] + u_now[i-1,j,k]
                    + u_now[i,j+1,k] + u_now[i,j-1,k]
                    + u_now[i,j,k+1] + u_now[i,j,k-1]
                    - 6.0*u_now[i,j,k]
                )
                src = (
                    g1[i,j,k]*np.cos(2*np.pi*t)
                    + g2[i,j,k]*np.cos(4*np.pi*t)
                    + g3[i,j,k]*tg
                )
                u_next[i,j,k] = 2*u_now[i,j,k] - u_prev[i,j,k] + coef*lap + dt*dt*src

spatial_slice = slice(None, None, 4)  # 401 -> 101
save_steps = list(range(0, nt+1, 3))  # 451 -> 151
out = np.empty((151,101,101,101), dtype=np.float64)
save_map = {s:i for i,s in enumerate(save_steps)}

tic = time.perf_counter()
for n in range(nt+1):
    if n in save_map:
        out[save_map[n]] = u_now[spatial_slice, spatial_slice, spatial_slice]
    if n == nt:
        break
    step(u_prev, u_now, u_next, g1, g2, g3, coef, dt, n*dt)
    u_next[0,:,:] = u_next[-1,:,:] = 0.0
    u_next[:,0,:] = u_next[:,-1,:] = 0.0
    u_next[:,:,0] = u_next[:,:,-1] = 0.0
    u_prev, u_now, u_next = u_now, u_next, u_prev
    u_next.fill(0.0)
    if n % 50 == 0:
        print(f"Step {n}/{nt}, elapsed {time.perf_counter()-tic:.1f} s")

output = DATA_DIR / "3d_fdtd_three_sources_2.5e-3.npy"
np.save(output, out)
print("Saved:", output, "shape=", out.shape)
print(f"Total runtime: {time.perf_counter()-tic:.3f} s")
