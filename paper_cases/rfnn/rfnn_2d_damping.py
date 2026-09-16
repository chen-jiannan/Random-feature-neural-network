"""Paper reproduction script for global random-feature wave experiments.

This file is a cleaned, English-only version of the corresponding code supplied with
the manuscript. GPU indices are not hard-coded; set CUDA_VISIBLE_DEVICES externally
if a specific GPU is desired.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# ========== Parameter Configuration ==========
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# =========================
# Parameters consistent with FDTD-PML absorbing boundary
# =========================
c = 2.0
L = 1.0
T = 1.0

# ---------- External PML absorbing layer ----------
pml_width = 1.5
sigma_max = 10.0
sigma_order = 3.0

# Extended computational domain
x_left = -pml_width
x_right = L + pml_width
y_bottom = -pml_width
y_top = L + pml_width

# ---------- Random feature parameters ----------
num_neuron = 10000
r_neuron = [5, 5, 5]   # t, x, y
d = 3

# ---------- Source parameters (consistent with FDTD) ----------
x_src, y_src = L / 2.0, L / 2.0
sigma_src = 0.3
omega = 2.0 * np.pi

# ---------- Collocation weights ----------
p_pde = 1.0
p_bnd = 10.0
p_ini_u = 10.0
p_ini_v = 10.0

# ---------- Training point counts ----------
N_pde = 20000
N_bnd = 20000
N_ini = 10000

# ---------- Fixed reference solution file ----------
reference_file = DATA_DIR / DATA_DIR / '2d_fdtd_ref_2e-4.npy'

# ---------- Probe points (consistent with previous FDTD detection aperture) ----------
probe_points = [
    (0.1, 0.5),
    (0.9, 0.5),
    (0.5, 0.1),
    (0.5, 0.9),
]

# ---------- Boundary band width ----------
boundary_band = 0.3

# ========== Activation function and derivatives ==========
def atv(x):
    return torch.tanh(x)

def diff1_atv(x):
    return 1.0 - torch.tanh(x) ** 2

def diff2_atv(x):
    th = torch.tanh(x)
    return 2.0 * th * (th**2 - 1.0)

# ========== Damping function sigma(x, y) ==========
def sigma_xy(X, Y):
    """
    In the main domain [0,L]x[0,L], sigma = 0
    In the outer absorbing layer, sigma increases polynomially
    """
    sig = torch.zeros_like(X, dtype=dtype, device=X.device)

    left_mask = X < 0.0
    if left_mask.any():
        sig[left_mask] += sigma_max * ((0.0 - X[left_mask]) / pml_width) ** sigma_order

    right_mask = X > L
    if right_mask.any():
        sig[right_mask] += sigma_max * ((X[right_mask] - L) / pml_width) ** sigma_order

    bottom_mask = Y < 0.0
    if bottom_mask.any():
        sig[bottom_mask] += sigma_max * ((0.0 - Y[bottom_mask]) / pml_width) ** sigma_order

    top_mask = Y > L
    if top_mask.any():
        sig[top_mask] += sigma_max * ((Y[top_mask] - L) / pml_width) ** sigma_order

    return sig

# ========== Training Data Generation ==========
def generate_points():
    # PDE internal points: uniform sampling over the extended domain
    ti = torch.rand(N_pde, 1, device=device, dtype=dtype) * T
    xi = x_left + (x_right - x_left) * torch.rand(N_pde, 1, device=device, dtype=dtype)
    yi = y_bottom + (y_top - y_bottom) * torch.rand(N_pde, 1, device=device, dtype=dtype)

    # Enhanced sampling near the source
    n_focus = N_pde // 100
    xi_focus = torch.normal(
        mean=x_src,
        std=0.3,
        size=(n_focus, 1),
        device=device,
        dtype=dtype
    ).clamp(x_left, x_right)

    yi_focus = torch.normal(
        mean=y_src,
        std=0.3,
        size=(n_focus, 1),
        device=device,
        dtype=dtype
    ).clamp(y_bottom, y_top)

    ti_focus = torch.rand(n_focus, 1, device=device, dtype=dtype) * T

    pti_uniform = torch.cat((ti, xi, yi), dim=1)
    pti_focus = torch.cat((ti_focus, xi_focus, yi_focus), dim=1)
    pti = torch.cat((pti_uniform, pti_focus), dim=0)

    # Outer boundary points: the four outermost edges of the extended domain
    tb = torch.rand(N_bnd, 1, device=device, dtype=dtype) * T
    xb = x_left + (x_right - x_left) * torch.rand(N_bnd, 1, device=device, dtype=dtype)
    yb = y_bottom + (y_top - y_bottom) * torch.rand(N_bnd, 1, device=device, dtype=dtype)

    ptb_x0 = torch.cat((tb, torch.full_like(tb, x_left), yb), dim=1)
    ptb_x1 = torch.cat((tb, torch.full_like(tb, x_right), yb), dim=1)
    ptb_y0 = torch.cat((tb, xb, torch.full_like(tb, y_bottom)), dim=1)
    ptb_y1 = torch.cat((tb, xb, torch.full_like(tb, y_top)), dim=1)

    ptb = torch.cat((ptb_x0, ptb_x1, ptb_y0, ptb_y1), dim=0)

    # Initial conditions: t=0 over the extended domain
    ti_ini = torch.zeros(N_ini, 1, device=device, dtype=dtype)
    xi_ini = x_left + (x_right - x_left) * torch.rand(N_ini, 1, device=device, dtype=dtype)
    yi_ini = y_bottom + (y_top - y_bottom) * torch.rand(N_ini, 1, device=device, dtype=dtype)
    pt_ini = torch.cat((ti_ini, xi_ini, yi_ini), dim=1)

    return pti, ptb, pt_ini

# ========== PDE Residual ==========
def wave_equation_residual(W, b, points, c):
    Tt = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]

    W_t = W[0:1, :]
    W_x = W[1:2, :]
    W_y = W[2:3, :]

    Z = Tt @ W_t + X @ W_x + Y @ W_y + b

    phi_t = W_t * diff1_atv(Z)
    phi_tt = (W_t ** 2) * diff2_atv(Z)
    phi_xx = (W_x ** 2) * diff2_atv(Z)
    phi_yy = (W_y ** 2) * diff2_atv(Z)

    sig = sigma_xy(X, Y)

    # u_tt + sigma*u_t - c^2(u_xx + u_yy)
    return phi_tt + sig * phi_t - (c ** 2) * (phi_xx + phi_yy)

# ========== Initial Velocity Residual ==========
def ini_t_residual(W, b, points):
    Tt = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]

    W_t = W[0:1, :]
    W_x = W[1:2, :]
    W_y = W[2:3, :]

    Z = Tt @ W_t + X @ W_x + Y @ W_y + b
    return W_t * diff1_atv(Z)

# ========== Source Term ==========
def source_term(points):
    Tt = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]

    return (
        torch.cos(omega * Tt) *
        torch.exp(-((X - x_src) ** 2 + (Y - y_src) ** 2) / (2.0 * sigma_src ** 2))
    )

# ========== Utility Function ==========
def get_nearest_idx_2d(xq, yq, x_grid, y_grid):
    ix = np.argmin(np.abs(x_grid - xq))
    iy = np.argmin(np.abs(y_grid - yq))
    return ix, iy

# ========== Main Program ==========
start_time = time.time()

# Initialize random feature parameters
W1 = torch.tensor(
    np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)),
    dtype=dtype, device=device
)
W2 = torch.tensor(
    np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)),
    dtype=dtype, device=device
)
W3 = torch.tensor(
    np.random.uniform(-r_neuron[2], r_neuron[2], (1, num_neuron)),
    dtype=dtype, device=device
)
W = torch.cat((W1, W2, W3), dim=0)

# Bias initialization: covers the extended domain
b1 = torch.tensor(
    np.random.uniform(0.0, T, (1, num_neuron)),
    dtype=dtype, device=device
)
b2 = torch.tensor(
    np.random.uniform(x_left, x_right, (1, num_neuron)),
    dtype=dtype, device=device
)
b3 = torch.tensor(
    np.random.uniform(y_bottom, y_top, (1, num_neuron)),
    dtype=dtype, device=device
)
b_shift = torch.cat((b1, b2, b3), dim=0)
b = -torch.sum(b_shift * W, dim=0, keepdim=True)

# Training points
pti, ptb, pt_ini = generate_points()

# Construct matrix blocks
A_pde = wave_equation_residual(W, b, pti, c)
A_bnd = atv(ptb @ W + b)                # u=0 on the outermost boundary of the extended domain
A_ini_u = atv(pt_ini @ W + b)           # u(x,y,0)=0
A_ini_v = ini_t_residual(W, b, pt_ini)  # u_t(x,y,0)=0

A = torch.cat([
    p_pde * A_pde,
    p_bnd * A_bnd,
    p_ini_u * A_ini_u,
    p_ini_v * A_ini_v
], dim=0)

F_pde = source_term(pti)
F_bnd = torch.zeros((ptb.shape[0], 1), dtype=dtype, device=device)
F_ini_u = torch.zeros((pt_ini.shape[0], 1), dtype=dtype, device=device)
F_ini_v = torch.zeros((pt_ini.shape[0], 1), dtype=dtype, device=device)

F = torch.cat([
    p_pde * F_pde,
    p_bnd * F_bnd,
    p_ini_u * F_ini_u,
    p_ini_v * F_ini_v
], dim=0)

# Least squares solution
Ur = torch.linalg.lstsq(A, F).solution

end_time = time.time()
print("Total_time:", end_time - start_time)

# ========== Load Fixed Reference Solution ==========
fdtd = np.load(reference_file).astype(np.float64)
print("Loaded fixed FDTD reference shape:", fdtd.shape)

nt_ref, nx_ref, ny_ref = fdtd.shape

# Predict only in the main domain for direct comparison with the fixed reference solution
t_test = torch.linspace(0.0, T, nt_ref, device=device, dtype=dtype)
x_test = torch.linspace(0.0, L, nx_ref, device=device, dtype=dtype)
y_test = torch.linspace(0.0, L, ny_ref, device=device, dtype=dtype)

T_mesh, X_mesh, Y_mesh = torch.meshgrid(t_test, x_test, y_test, indexing='ij')
test_points = torch.cat(
    (
        T_mesh.reshape(-1, 1),
        X_mesh.reshape(-1, 1),
        Y_mesh.reshape(-1, 1)
    ),
    dim=1
)

# ========== Batch Prediction ==========
batch_size = 40000
u_pred = []

with torch.no_grad():
    for i in range(0, test_points.shape[0], batch_size):
        batch = test_points[i:i + batch_size]
        hidden = torch.tanh(batch @ W + b)
        output = hidden @ Ur
        u_pred.append(output.cpu())

u_pred = torch.cat(u_pred, dim=0).numpy().reshape(nt_ref, nx_ref, ny_ref)

# ========== Error ==========
error = u_pred - fdtd
rel_err = np.linalg.norm(error.ravel()) / np.linalg.norm(fdtd.ravel())
print(f"Relative L2 Error: {rel_err:.2%}")

# ============================================================
# Deviation curves at four probe points + Normalized deviation metric
# ============================================================
probe_results = []

print("\n===== Probe-point normalized deviation indicators =====")
x_grid = np.linspace(0.0, L, nx_ref)
y_grid = np.linspace(0.0, L, ny_ref)
t_grid = np.linspace(0.0, T, nt_ref)

for (xp, yp) in probe_points:
    ix, iy = get_nearest_idx_2d(xp, yp, x_grid, y_grid)

    trace_num = u_pred[:, ix, iy]
    trace_ref = fdtd[:, ix, iy]
    trace_err = trace_num - trace_ref

    numerator = np.max(np.abs(trace_err))
    denominator = np.max(np.abs(trace_ref))
    D = numerator / max(denominator, 1e-15)
    D_dB = 20.0 * np.log10(max(D, 1e-15))

    probe_results.append({
        "x": x_grid[ix],
        "y": y_grid[iy],
        "ix": ix,
        "iy": iy,
        "trace_num": trace_num,
        "trace_ref": trace_ref,
        "trace_err": trace_err,
        "D": D,
        "D_dB": D_dB
    })

    print(f"(x, y)=({x_grid[ix]:.2f}, {y_grid[iy]:.2f}) : D = {D:.6e}, D_dB = {D_dB:.2f} dB")

# ============================================================
# Normalized deviation metric near boundaries
# ============================================================
xv, yv = np.meshgrid(x_grid, y_grid, indexing='ij')
boundary_mask = (
    (xv <= boundary_band) |
    (xv >= L - boundary_band) |
    (yv <= boundary_band) |
    (yv >= L - boundary_band)
)

D_map = np.zeros((nx_ref, ny_ref), dtype=np.float64)
D_map_dB = np.zeros((nx_ref, ny_ref), dtype=np.float64)

for i in range(nx_ref):
    for j in range(ny_ref):
        ref_trace = fdtd[:, i, j]
        err_trace = error[:, i, j]

        numerator = np.max(np.abs(err_trace))
        denominator = np.max(np.abs(ref_trace))
        D_local = numerator / max(denominator, 1e-15)

        D_map[i, j] = D_local
        D_map_dB[i, j] = 20.0 * np.log10(max(D_local, 1e-15))

boundary_D_values = D_map[boundary_mask]
boundary_D_dB_values = D_map_dB[boundary_mask]

print("\n===== Boundary-band normalized deviation metrics =====")
print(f"Boundary band width = {boundary_band}")
print(f"Max   D in boundary band    = {np.max(boundary_D_values):.6e}")
print(f"Mean  D in boundary band    = {np.mean(boundary_D_values):.6e}")
print(f"Max   D_dB in boundary band = {np.max(boundary_D_dB_values):.2f} dB")
print(f"Mean  D_dB in boundary band = {np.mean(boundary_D_dB_values):.2f} dB")

# ============================================================
# Dynamic time slices
# ============================================================
target_times_ratio = [0.25, 0.5, 0.75, 1.0]
target_times = np.array(target_times_ratio) * T
time_indices = np.floor(np.array(target_times_ratio) * (nt_ref - 1)).astype(int)

# ============================================================
# Visualization 1: Main domain snapshots
# ============================================================
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
        ax.set_title(f"t = {target_times[idx]:.2f}s", fontweight='bold', fontsize=14)
        ax.set_xlabel("x", fontsize=12, fontweight='bold')
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_snapshots(u_pred, "2D RFNN Wave Field with Damping Absorbing Layers")
plot_snapshots(fdtd, "2D Fixed FDTD Reference Field")
plot_snapshots(error, "2D Deviation Field: RFNN - Reference")

# ============================================================
# Visualization 2: Main domain deviation space-time plot
# Taking the y = L/2 midline
# ============================================================
iy_mid = np.argmin(np.abs(y_grid - L / 2.0))
error_xt = error[:, :, iy_mid]

X_plot, T_plot = np.meshgrid(x_grid, t_grid)

plt.figure(figsize=(10, 6))
plt.pcolormesh(X_plot, T_plot, error_xt, shading='auto', cmap='RdBu')
plt.colorbar(label='Deviation = RFNN - Reference')
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'Deviation space-time map on center line y = {y_grid[iy_mid]:.2f}')
plt.tight_layout()
plt.show()

# ============================================================
# Visualization 3: Deviation curves at four probe points
# ============================================================
plt.figure(figsize=(12, 8))
for k, item in enumerate(probe_results):
    plt.subplot(2, 2, k + 1)
    plt.plot(t_grid, item["trace_err"], linewidth=2, label='Deviation')
    plt.plot(t_grid, item["trace_num"], '--', linewidth=1.5, label='RFNN')
    plt.plot(t_grid, item["trace_ref"], ':', linewidth=1.5, label='Reference')
    plt.xlabel('t')
    plt.ylabel('Field')
    plt.title(
        f"(x,y)=({item['x']:.2f},{item['y']:.2f})\n"
        f"D = {item['D']:.3e}, D_dB = {item['D_dB']:.2f} dB"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()
