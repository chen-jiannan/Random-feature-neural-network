"""Generate the 1D homogeneous FDTD reference used by the RFNN/PINN scripts."""
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
    parser.add_argument("--dx", type=float, default=1e-4)
    parser.add_argument("--save-dx", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    c, L, T = 1.0, 5.0, 5.0
    dx = args.dx
    dt = dx / c
    nx = int(round(L / dx)) + 1
    nt = int(round(T / dt))

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = torch.float64
    print("Device:", device, "| nx:", nx, "| nt:", nt + 1)

    x = torch.linspace(0.0, L, nx, device=device, dtype=dtype)
    g1 = torch.exp(-((x - 1.5) ** 2) / (2 * 0.1 ** 2))
    g2 = torch.exp(-((x - 4.0) ** 2) / (2 * 0.1 ** 2))

    u_prev = torch.zeros(nx, device=device, dtype=dtype)
    u_now = torch.zeros_like(u_prev)
    u_next = torch.zeros_like(u_prev)

    save_stride_x = max(1, int(round(args.save_dx / dx)))
    save_stride_t = max(1, int(round(args.save_dx / (c * dt))))
    x_idx = torch.arange(0, nx, save_stride_x, device=device)
    save_steps = list(range(0, nt + 1, save_stride_t))
    if save_steps[-1] != nt:
        save_steps.append(nt)

    out = np.empty((len(save_steps), len(x_idx)), dtype=np.float64)
    save_map = {step: i for i, step in enumerate(save_steps)}

    if device.type == "cuda":
        torch.cuda.synchronize()
    tic = time.perf_counter()

    coef = (c * dt / dx) ** 2
    for n in range(nt + 1):
        if n in save_map:
            out[save_map[n]] = u_now.index_select(0, x_idx).cpu().numpy()
        if n == nt:
            break

        t = n * dt
        u_next[1:-1] = (
            2.0 * u_now[1:-1] - u_prev[1:-1]
            + coef * (u_now[2:] - 2.0 * u_now[1:-1] + u_now[:-2])
        )
        u_next += (dt ** 2) * (
            2.0 * g1 * torch.cos(torch.tensor(2*np.pi*t, device=device, dtype=dtype))
            + 5.0 * g2 * torch.cos(torch.tensor(4*np.pi*t, device=device, dtype=dtype))
        )
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u_prev, u_now, u_next = u_now, u_next, u_prev
        u_next.zero_()

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"FDTD runtime: {time.perf_counter() - tic:.3f} s")

    output = DATA_DIR / "1d_fdtd_two_sources_cos_1e-4.npy"
    np.save(output, out)
    print("Saved:", output, "shape=", out.shape)

if __name__ == "__main__":
    main()
