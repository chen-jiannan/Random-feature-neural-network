"""Generate the 2D homogeneous FDTD reference on the paper evaluation grid."""
from pathlib import Path
import argparse
import time
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    c, L, T = 1.0, 5.0, 1.0
    dx = 2e-3
    nx = int(round(L / dx)) + 1

    # Choose nt=750 so the CFL number is below one and every fifth step
    # maps exactly to the 151-point paper evaluation time grid.
    nt = 750
    dt = T / nt
    cfl = c * dt * np.sqrt(2.0) / dx
    if cfl > 1.0:
        raise RuntimeError(f"CFL condition violated: {cfl:.6f}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = torch.float64
    print("Device:", device, "| grid:", nx, "x", nx, "| nt:", nt + 1, "| CFL:", cfl)

    x = torch.linspace(0.0, L, nx, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    g1 = torch.exp(-((X - 1.5)**2 + (Y - 1.5)**2) / (2 * 0.2**2))
    g2 = torch.exp(-((X - 4.0)**2 + (Y - 4.0)**2) / (2 * 0.2**2))

    u_prev = torch.zeros((nx, nx), device=device, dtype=dtype)
    u_now = torch.zeros_like(u_prev)
    u_next = torch.zeros_like(u_prev)
    coef = (c * dt / dx) ** 2

    spatial_idx = torch.arange(0, nx, 5, device=device)  # 501 points
    save_steps = list(range(0, nt + 1, 5))              # 151 points
    out = np.empty((len(save_steps), len(spatial_idx), len(spatial_idx)), dtype=np.float64)
    save_map = {s: i for i, s in enumerate(save_steps)}

    if device.type == "cuda":
        torch.cuda.synchronize()
    tic = time.perf_counter()

    for n in range(nt + 1):
        if n in save_map:
            sample = u_now.index_select(0, spatial_idx).index_select(1, spatial_idx)
            out[save_map[n]] = sample.cpu().numpy()
        if n == nt:
            break

        t = n * dt
        lap = (
            u_now[2:,1:-1] + u_now[:-2,1:-1]
            + u_now[1:-1,2:] + u_now[1:-1,:-2]
            - 4.0*u_now[1:-1,1:-1]
        )
        u_next[1:-1,1:-1] = (
            2.0*u_now[1:-1,1:-1] - u_prev[1:-1,1:-1] + coef*lap
        )
        source = (
            g1 * np.cos(2*np.pi*t)
            + g2 * np.exp(-((t - 0.1)**2)/(2*0.1**2))
        )
        u_next += (dt**2) * source
        u_next[0,:] = u_next[-1,:] = 0.0
        u_next[:,0] = u_next[:,-1] = 0.0
        u_prev, u_now, u_next = u_now, u_next, u_prev
        u_next.zero_()

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"FDTD runtime: {time.perf_counter() - tic:.3f} s")

    output = DATA_DIR / "2d_fdtd_two_source_2e-3.npy"
    np.save(output, out)
    print("Saved:", output, "shape=", out.shape)

if __name__ == "__main__":
    main()
