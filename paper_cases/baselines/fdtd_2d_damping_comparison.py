"""2D damping-layer FDTD comparison from the manuscript workflow.
Requires data/2d_fdtd_ref_2e-4.npy, which is not bundled because of its size.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

c = 2.0
L = 1.0
T = 1.0

dx = 5e-4
dy = dx

pml_width = 1.5
n_pml = int(np.round(pml_width / dx))
sigma_max = 10.0
sigma_order = 3.0

# ---------- CFL ----------
CFL = 0.9
dt = CFL / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
nt = int(np.ceil(T / dt))
dt = T / nt

CFL_actual = c * dt * np.sqrt(1.0 / dx**2 + 1.0 / dy**2)
if CFL_actual > 1.0 + 1e-12:
    raise ValueError(f"CFL violated: {CFL_actual:.6f} > 1.0")

nx_total = int(round(L / dx)) + 1
ny_total = int(round(L / dy)) + 1

nx = nx_total + 2 * n_pml
ny = ny_total + 2 * n_pml

dx_save = 0.01
dy_save = dx_save

dt_save = CFL / (c * np.sqrt(1.0 / dx_save**2 + 1.0 / dy_save**2))
nt_save = int(round(T / dt_save)) + 1

nx_save = int(round(L / dx_save)) + 1
ny_save = int(round(L / dy_save)) + 1

x_save_1d = np.linspace(0.0, L, nx_save)
y_save_1d = np.linspace(0.0, L, ny_save)
t_save_1d = np.linspace(0.0, T, nt_save)

x_src, y_src = L / 2, L / 2
sigma_src = 0.3
omega = 2.0 * np.pi

reference_file = DATA_DIR / DATA_DIR / '2d_fdtd_ref_2e-4.npy'

probe_points = [
    (0.1, 0.5),
    (0.5, 0.1),
    (0.9, 0.5),
    (0.5, 0.9),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(f"Using device: {device}")

x_1d = (torch.arange(nx, device=device, dtype=dtype) - n_pml) * dx
y_1d = (torch.arange(ny, device=device, dtype=dtype) - n_pml) * dy
X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')

gaussian_source_weights = torch.exp(
    -((X - x_src) ** 2 + (Y - y_src) ** 2) / (2 * sigma_src ** 2)
)

sigma = torch.zeros((nx, ny), device=device, dtype=dtype)

for i in range(n_pml):
    x_norm = (n_pml - i) / n_pml
    sigma[i, :] += sigma_max * (x_norm ** sigma_order)
for i in range(nx - n_pml, nx):
    x_norm = (i - (nx - n_pml - 1)) / n_pml
    sigma[i, :] += sigma_max * (x_norm ** sigma_order)

for j in range(n_pml):
    y_norm = (n_pml - j) / n_pml
    sigma[:, j] += sigma_max * (y_norm ** sigma_order)
for j in range(ny - n_pml, ny):
    y_norm = (j - (ny - n_pml - 1)) / n_pml
    sigma[:, j] += sigma_max * (y_norm ** sigma_order)

denom = 1.0 + sigma * dt / 2.0
C1 = 2.0 / denom
C2 = (1.0 - sigma * dt / 2.0) / denom
C3_x = ((c * dt / dx) ** 2) / denom
C3_y = ((c * dt / dy) ** 2) / denom
C_src = (dt ** 2) / denom

x_save_idx = np.round(x_save_1d / dx).astype(int) + n_pml
y_save_idx = np.round(y_save_1d / dy).astype(int) + n_pml
t_save_idx = np.round(t_save_1d / dt).astype(int)

x_save_idx = np.clip(x_save_idx, 0, nx - 1)
y_save_idx = np.clip(y_save_idx, 0, ny - 1)
t_save_idx = np.clip(t_save_idx, 0, nt)

x_save_idx = np.unique(x_save_idx)
y_save_idx = np.unique(y_save_idx)
t_save_idx = np.unique(t_save_idx)

nx_save = len(x_save_idx)
ny_save = len(y_save_idx)
nt_save = len(t_save_idx)

x_save_1d = np.linspace(0.0, L, nx_save)
y_save_1d = np.linspace(0.0, L, ny_save)
t_save_1d = np.linspace(0.0, T, nt_save)

x_save_idx_t = torch.tensor(x_save_idx, device=device, dtype=torch.long)
y_save_idx_t = torch.tensor(y_save_idx, device=device, dtype=torch.long)

print("=" * 60)
print("Damping-layer computation:")
print(f"dx = {dx:.6f}, dt = {dt:.6f}")
print(f"CFL actual = {CFL_actual:.6f}")
print(f"Main domain cells: {nx_total} x {ny_total}")
print(f"Damping cells per boundary: {n_pml}")
print(f"Total cells: nx = {nx}, ny = {ny}, nt = {nt+1}")
print(f"Saved main-domain data shape = ({nt_save}, {nx_save}, {ny_save})")
print("=" * 60)

E_prev = torch.zeros((nx, ny), device=device, dtype=dtype)
E_current = torch.zeros((nx, ny), device=device, dtype=dtype)
E_next = torch.zeros((nx, ny), device=device, dtype=dtype)

E_history_save = np.zeros((nt_save, nx_save, ny_save), dtype=np.float64)

if device.type.startswith("cuda"):
    torch.cuda.synchronize()

total_start = time.perf_counter()
loop_start = time.perf_counter()

save_counter = 0

for n in range(nt + 1):
    E_next[1:-1, 1:-1] = (
        C1[1:-1, 1:-1] * E_current[1:-1, 1:-1]
        - C2[1:-1, 1:-1] * E_prev[1:-1, 1:-1]
        + C3_x[1:-1, 1:-1] * (
            E_current[2:, 1:-1]
            - 2.0 * E_current[1:-1, 1:-1]
            + E_current[:-2, 1:-1]
        )
        + C3_y[1:-1, 1:-1] * (
            E_current[1:-1, 2:]
            - 2.0 * E_current[1:-1, 1:-1]
            + E_current[1:-1, :-2]
        )
    )

    t_now = n * dt
    amplitude = 0.5 if n < 1 else 1.0
    source_factor = amplitude * np.cos(omega * t_now)
    E_next += gaussian_source_weights * source_factor * C_src

    E_next[0, :] = 0.0
    E_next[-1, :] = 0.0
    E_next[:, 0] = 0.0
    E_next[:, -1] = 0.0

    if save_counter < nt_save and n == t_save_idx[save_counter]:
        sampled = E_current.index_select(0, x_save_idx_t).index_select(1, y_save_idx_t)
        E_history_save[save_counter] = sampled.detach().cpu().numpy()
        save_counter += 1

    E_prev, E_current, E_next = E_current, E_next, E_prev
    E_next.zero_()

if device.type.startswith("cuda"):
    torch.cuda.synchronize()

loop_end = time.perf_counter()
total_end = time.perf_counter()

print(f"Pure FDTD loop time: {loop_end - loop_start:.6f} s")
print(f"Total time: {total_end - total_start:.6f} s")

# ============================================================
# ============================================================
E_ref = np.load(reference_file).astype(np.float64)
print(f"Loaded reference file: {reference_file}")
print(f"Reference shape: {E_ref.shape}")

if E_ref.shape != E_history_save.shape:
    raise ValueError(
        f"Reference shape {E_ref.shape} != damping-result shape {E_history_save.shape}"
    )

# ============================================================
# ============================================================
E_error = E_history_save - E_ref

rel_l2_error = np.linalg.norm(E_error.ravel()) / max(np.linalg.norm(E_ref.ravel()), 1e-15)
print(f"Relative L2 Error (main domain): {rel_l2_error:.2%}")

# ============================================================
# ============================================================
def get_nearest_idx_2d(xq, yq, x_grid, y_grid):
    ix = np.argmin(np.abs(x_grid - xq))
    iy = np.argmin(np.abs(y_grid - yq))
    return ix, iy

probe_results = []

print("\n===== Probe-point normalized reflection indicators =====")
for (xp, yp) in probe_points:
    ix, iy = get_nearest_idx_2d(xp, yp, x_save_1d, y_save_1d)

    trace_num = E_history_save[:, ix, iy]
    trace_ref = E_ref[:, ix, iy]
    trace_err = trace_num - trace_ref

    numerator = np.max(np.abs(trace_err))
    denominator = np.max(np.abs(trace_ref))
    R = numerator / max(denominator, 1e-15)
    R_dB = 20.0 * np.log10(max(R, 1e-15))

    probe_results.append({
        "x": x_save_1d[ix],
        "y": y_save_1d[iy],
        "ix": ix,
        "iy": iy,
        "trace_num": trace_num,
        "trace_ref": trace_ref,
        "trace_err": trace_err,
        "R": R,
        "R_dB": R_dB
    })

    print(f"(x, y)=({x_save_1d[ix]:.2f}, {y_save_1d[iy]:.2f}) : R = {R:.6e}, R_dB = {R_dB:.2f} dB")

# ============================================================
# ============================================================
boundary_band = 0.3

xv, yv = np.meshgrid(x_save_1d, y_save_1d, indexing='ij')
boundary_mask = (
    (xv <= boundary_band) |
    (xv >= L - boundary_band) |
    (yv <= boundary_band) |
    (yv >= L - boundary_band)
)

R_map = np.zeros((nx_save, ny_save), dtype=np.float64)
R_map_dB = np.zeros((nx_save, ny_save), dtype=np.float64)

for i in range(nx_save):
    for j in range(ny_save):
        ref_trace = E_ref[:, i, j]
        err_trace = E_error[:, i, j]

        numerator = np.max(np.abs(err_trace))
        denominator = np.max(np.abs(ref_trace))
        R_local = numerator / max(denominator, 1e-15)
        R_map[i, j] = R_local
        R_map_dB[i, j] = 20.0 * np.log10(max(R_local, 1e-15))

boundary_R_values = R_map[boundary_mask]
boundary_R_dB_values = R_map_dB[boundary_mask]

print("\n===== Boundary-band normalized reflection metrics =====")
print(f"Boundary band width = {boundary_band}")
print(f"Max   R in boundary band   = {np.max(boundary_R_values):.6e}")
print(f"Mean  R in boundary band   = {np.mean(boundary_R_values):.6e}")
print(f"Max   R_dB in boundary band = {np.max(boundary_R_dB_values):.2f} dB")
print(f"Mean  R_dB in boundary band = {np.mean(boundary_R_dB_values):.2f} dB")

# ============================================================
# ============================================================
target_times_ratio = [0.25, 0.5, 0.75, 1.0]
target_times = np.array(target_times_ratio) * T
time_indices = [np.argmin(np.abs(t_save_1d - tt)) for tt in target_times]

def plot_snapshots(field_data, title):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    vmax = np.max(np.abs(field_data))
    vmin = -vmax

    for idx, (ax, t_idx) in enumerate(zip(axs.flat, time_indices)):
        im = ax.imshow(
            field_data[t_idx].T,
            extent=[0, L, 0, L],
            origin='lower',
            cmap='jet',
            aspect='auto'
        )
        ax.set_title(f"t = {t_save_1d[t_idx]:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

plot_snapshots(E_history_save, "Wave Field in Main Domain with Damping Layer")
plot_snapshots(E_ref, "Reflection-free Reference Field in Main Domain")

# ============================================================
# ============================================================
iy_mid = np.argmin(np.abs(y_save_1d - L / 2))
error_xt = E_error[:, :, iy_mid]   # shape = (nt_save, nx_save)

X_plot, T_plot = np.meshgrid(x_save_1d, t_save_1d)

plt.figure(figsize=(10, 6))
plt.pcolormesh(X_plot, T_plot, error_xt, shading='auto', cmap='RdBu')
plt.colorbar(label='Error = E_damp - E_ref')
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'Error space-time map on center line y = {y_save_1d[iy_mid]:.2f}')
plt.tight_layout()
plt.show()

# ============================================================
# ============================================================
plt.figure(figsize=(12, 8))
for k, item in enumerate(probe_results):
    plt.subplot(2, 2, k + 1)
    plt.plot(t_save_1d, item["trace_err"], linewidth=2, label='Error')
    plt.plot(t_save_1d, item["trace_num"], '--', linewidth=1.5, label='Damping')
    plt.plot(t_save_1d, item["trace_ref"], ':', linewidth=1.5, label='Reference')
    plt.xlabel('t')
    plt.ylabel('Field')
    plt.title(
        f"(x,y)=({item['x']:.2f},{item['y']:.2f})\n"
        f"R = {item['R']:.3e}, R_dB = {item['R_dB']:.2f} dB"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# ============================================================
plt.figure(figsize=(8, 6))
im = plt.imshow(
    R_map_dB.T,
    extent=[0, L, 0, L],
    origin='lower',
    cmap='jet',
    aspect='auto'
)
plt.colorbar(im, label='Normalized Reflection Indicator (dB)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spatial distribution of normalized reflection indicator (dB)')
plt.contour(x_save_1d, y_save_1d, boundary_mask.T.astype(float), levels=[0.5], colors='w', linewidths=1.5)
plt.tight_layout()
plt.show()
