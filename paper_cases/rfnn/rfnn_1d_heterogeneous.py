"""Paper reproduction script for global random-feature wave experiments.

This file is a cleaned, English-only version of the corresponding code supplied with
the manuscript. GPU indices are not hard-coded; set CUDA_VISIBLE_DEVICES externally
if a specific GPU is desired.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from numba import njit

# ============================================================
# Output directory
# ============================================================
output_dir = "nonuniform_1d_combined"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Unified plot style
# ============================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def save_fig(fig, filename_base):
    png = os.path.join(output_dir, f"{filename_base}.png")
    pdf = os.path.join(output_dir, f"{filename_base}.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

paper_colors = {
    "ref": "#d62728",
    "fdtd": "#2ca02c",
    "rfnn": "#1f77b4",
    "gray": "#7f7f7f",
}

# ============================================================
# Global parameters (some will be changed per beta)
# ============================================================
initial_seed = 5
torch.manual_seed(initial_seed)
np.random.seed(initial_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# -----------------------------
# Physical domain
# -----------------------------
L = 5.0
T = 5.0

# -----------------------------
# Stronger nonuniform medium
# Smooth slab-type low-speed region
# -----------------------------
c0 = 1.0
beta = 0.18   # will be overwritten in the beta study loop
x1 = 2.0
x2 = 3.0
w_if = 0.10
x_src = 1.2

# -----------------------------
# Source (moved to left side)
# -----------------------------
source_sigma = 0.10
freq = 1.0
omega = 2.0 * np.pi * freq

# -----------------------------
# RFNN parameters
# -----------------------------
num_neuron = 3000
r_neuron = [5, 5]

N_pde = 30000
N_bnd = 20000
N_ini = 10000

p_pde = 1.0
p_bnd = 10.0
p_ini = 10.0

solve_block_size = 5000
predict_chunk_size = 20

# -----------------------------
# FDTD settings
# -----------------------------
dx_ref = 5e-5     # fine-grid reference
dx_fdtd = 1e-2    # coarse-grid FDTD baseline
CFL = 0.9

# -----------------------------
# Evaluation grid for comparison / paper figures
# -----------------------------
nx_eval = 101
nt_eval = 101
x_eval = np.linspace(0.0, L, nx_eval)
t_eval = np.linspace(0.0, T, nt_eval)

probe_positions = [0.8, 1.6, 3.8]   # 3 probe points for article figures

print(f"Using device: {device}")
print(f"Using dtype  : {dtype}")

# ============================================================
# Medium profile
# c(x) = c0 * (1 - beta * window(x))
# q(x) = c(x)^2
# ============================================================
def smooth_window_np(x):
    return 0.5 * (np.tanh((x - x1) / w_if) - np.tanh((x - x2) / w_if))

def c_profile_np(x):
    return c0 * (1.0 - beta * smooth_window_np(x))

def q_profile_np(x):
    c = c_profile_np(x)
    return c ** 2

def smooth_window_torch(x):
    return 0.5 * (torch.tanh((x - x1) / w_if) - torch.tanh((x - x2) / w_if))

def c_profile_torch(x):
    return c0 * (1.0 - beta * smooth_window_torch(x))

def q_profile_torch(x):
    c = c_profile_torch(x)
    return c ** 2

def dqdx_profile_torch(x):
    # derivative of q(x)=c(x)^2
    sech2_x1 = 1.0 / torch.cosh((x - x1) / w_if) ** 2
    sech2_x2 = 1.0 / torch.cosh((x - x2) / w_if) ** 2
    dwindow = 0.5 * (sech2_x1 / w_if - sech2_x2 / w_if)
    c = c_profile_torch(x)
    dc_dx = -c0 * beta * dwindow
    return 2.0 * c * dc_dx

# ============================================================
# Shared helpers
# ============================================================
def interpolate_trace_along_x(x_grid, u_xt, x_query):
    if x_query <= x_grid[0]:
        return u_xt[0, :].copy()
    if x_query >= x_grid[-1]:
        return u_xt[-1, :].copy()

    i1 = np.searchsorted(x_grid, x_query)
    i0 = i1 - 1
    x0 = x_grid[i0]
    x1_ = x_grid[i1]
    alpha_ = (x_query - x0) / (x1_ - x0)
    return (1.0 - alpha_) * u_xt[i0, :] + alpha_ * u_xt[i1, :]

def relative_l2(u_num_xt, u_ref_xt):
    return np.linalg.norm(u_num_xt - u_ref_xt) / max(np.linalg.norm(u_ref_xt), 1e-15)

# ============================================================
# FDTD solver for variable-coefficient wave equation
# u_tt = d/dx(q(x) u_x) + f(x,t)
# PEC: u(0,t)=u(L,t)=0
# ============================================================
@njit
def nearest_index_array(grid, values):
    idx = np.empty(values.shape[0], dtype=np.int64)
    for k in range(values.shape[0]):
        best_i = 0
        best_d = abs(grid[0] - values[k])
        for i in range(1, grid.shape[0]):
            d = abs(grid[i] - values[k])
            if d < best_d:
                best_d = d
                best_i = i
        idx[k] = best_i
    return idx

@njit
def run_fdtd_variable_q_sampled(
    x, q, source_space, dt, nt,
    save_t_idx, save_x_idx, probe_idx, omega
):
    nx = x.shape[0]
    dx = x[1] - x[0]

    u_prev = np.zeros(nx)
    u_prev[1:-1] = 0.5 * dt**2 * source_space[1:-1]
    u_curr = np.zeros(nx)
    u_next = np.zeros(nx)

    q_half = 0.5 * (q[:-1] + q[1:])

    n_save_t = save_t_idx.shape[0]
    n_save_x = save_x_idx.shape[0]
    n_probe = probe_idx.shape[0]

    u_save = np.zeros((n_save_t, n_save_x))
    probe_traces = np.zeros((n_probe, nt))

    save_counter = 0

    for n in range(nt):
        t_now = n * dt

        flux_right = q_half[1:] * (u_curr[2:] - u_curr[1:-1])
        flux_left  = q_half[:-1] * (u_curr[1:-1] - u_curr[:-2])
        spatial_op = (flux_right - flux_left) / (dx ** 2)

        source_t = np.cos(omega * t_now)

        u_next[1:-1] = (
            2.0 * u_curr[1:-1]
            - u_prev[1:-1]
            + (dt ** 2) * spatial_op
            + (dt ** 2) * source_space[1:-1] * source_t
        )

        u_next[0] = 0.0
        u_next[-1] = 0.0

        for p in range(n_probe):
            probe_traces[p, n] = u_curr[probe_idx[p]]

        if save_counter < n_save_t and n == save_t_idx[save_counter]:
            for j in range(n_save_x):
                u_save[save_counter, j] = u_curr[save_x_idx[j]]
            save_counter += 1

        tmp = u_prev
        u_prev = u_curr
        u_curr = u_next
        u_next = tmp

        for i in range(nx):
            u_next[i] = 0.0

    return u_save, probe_traces

def run_fdtd_case(dx, case_name):
    x = np.linspace(0.0, L, int(round(L / dx)) + 1)
    c_x = c_profile_np(x)
    q_x = c_x ** 2

    c_max = np.max(c_x)
    dt = CFL * dx / c_max
    nt = int(np.ceil(T / dt)) + 1
    dt = T / (nt - 1)

    t = np.linspace(0.0, T, nt)
    source_space = np.exp(-((x - x_src) ** 2) / (2.0 * source_sigma ** 2))

    save_t_idx = nearest_index_array(t, t_eval)
    save_x_idx = nearest_index_array(x, x_eval)

    probe_positions_np = np.array(probe_positions, dtype=np.float64)
    probe_idx = nearest_index_array(x, probe_positions_np)

    start = time.perf_counter()
    u_save, probe_traces = run_fdtd_variable_q_sampled(
        x=x,
        q=q_x,
        source_space=source_space,
        dt=dt,
        nt=nt,
        save_t_idx=save_t_idx,
        save_x_idx=save_x_idx,
        probe_idx=probe_idx,
        omega=omega
    )
    elapsed = time.perf_counter() - start

    print(f"{case_name}: dx={dx:.5e}, dt={dt:.5e}, nx={len(x)}, nt={nt}, time={elapsed:.3f}s")
    return x, t, c_x, q_x, u_save, probe_traces

# ============================================================
# RFNN
# PDE: u_tt - (q_x u_x + q u_xx) = f
# ============================================================
def atv(x):
    return torch.tanh(x)

def diff1_atv(x):
    return 1.0 - torch.tanh(x) ** 2

def diff2_atv(x):
    th = torch.tanh(x)
    return 2.0 * th * (th**2 - 1.0)

def generate_points():
    N_uniform = 18000
    N_src = 6000
    N_if1 = 3000
    N_if2 = N_pde - N_uniform - N_src - N_if1

    xi_uniform = torch.rand(N_uniform, 1, device=device, dtype=dtype) * L
    ti_uniform = torch.rand(N_uniform, 1, device=device, dtype=dtype) * T

    xi_src = torch.normal(
        mean=x_src, std=0.12, size=(N_src, 1), device=device
    ).to(dtype).clamp(0.0, L)
    ti_src = torch.rand(N_src, 1, device=device, dtype=dtype) * T

    xi_if1 = torch.normal(
        mean=x1, std=0.10, size=(N_if1, 1), device=device
    ).to(dtype).clamp(0.0, L)
    ti_if1 = torch.rand(N_if1, 1, device=device, dtype=dtype) * T

    xi_if2 = torch.normal(
        mean=x2, std=0.10, size=(N_if2, 1), device=device
    ).to(dtype).clamp(0.0, L)
    ti_if2 = torch.rand(N_if2, 1, device=device, dtype=dtype) * T

    xi = torch.cat([xi_uniform, xi_src, xi_if1, xi_if2], dim=0)
    ti = torch.cat([ti_uniform, ti_src, ti_if1, ti_if2], dim=0)
    pti = torch.cat((xi, ti), dim=1)

    tb = torch.rand(N_bnd, 1, device=device, dtype=dtype) * T
    ptb_0 = torch.cat((torch.zeros_like(tb), tb), dim=1)
    ptb_L = torch.cat((torch.full_like(tb, L), tb), dim=1)
    ptb = torch.cat((ptb_0, ptb_L), dim=0)

    xi_ini = torch.rand(N_ini, 1, device=device, dtype=dtype) * L
    ti_ini = torch.zeros(N_ini, 1, device=device, dtype=dtype)
    pt_ini = torch.cat((xi_ini, ti_ini), dim=1)

    return pti, ptb, pt_ini

def feature_value(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return atv(Z)

def feature_x(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return W_x * diff1_atv(Z)

def feature_t(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return W_t * diff1_atv(Z)

def feature_xx(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return (W_x ** 2) * diff2_atv(Z)

def feature_tt(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return (W_t ** 2) * diff2_atv(Z)

def source_term(points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    return torch.cos(omega * Tt) * torch.exp(-((X - x_src) ** 2) / (2.0 * source_sigma ** 2))

def wave_equation_residual(W, b, points):
    phi_tt = feature_tt(W, b, points)
    phi_x = feature_x(W, b, points)
    phi_xx = feature_xx(W, b, points)

    X = points[:, 0:1]
    qx = q_profile_torch(X)
    dqdx = dqdx_profile_torch(X)

    return phi_tt - (dqdx * phi_x + qx * phi_xx)

def ini_residual(W, b, points):
    return feature_t(W, b, points)

@torch.no_grad()
def solve_blockwise_normal_equation(W, b, pti, ptb, pt_ini, ridge=1e-6):
    M = W.shape[1]
    ATA = torch.zeros((M, M), dtype=dtype, device=device)
    ATF = torch.zeros((M, 1), dtype=dtype, device=device)

    n_pde_total = pti.shape[0]

    for i0 in range(0, n_pde_total, solve_block_size):
        i1 = min(i0 + solve_block_size, n_pde_total)
        pti_block = pti[i0:i1]

        A_pde_block = p_pde * wave_equation_residual(W, b, pti_block)
        F_pde_block = p_pde * source_term(pti_block)

        ATA += A_pde_block.T @ A_pde_block
        ATF += A_pde_block.T @ F_pde_block

        del pti_block, A_pde_block, F_pde_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    A_bnd = p_bnd * feature_value(W, b, ptb)
    F_bnd = torch.zeros((ptb.shape[0], 1), dtype=dtype, device=device)
    ATA += A_bnd.T @ A_bnd
    ATF += A_bnd.T @ F_bnd
    del A_bnd, F_bnd

    A_ini1 = p_ini * feature_value(W, b, pt_ini)
    F_ini1 = torch.zeros((pt_ini.shape[0], 1), dtype=dtype, device=device)
    ATA += A_ini1.T @ A_ini1
    ATF += A_ini1.T @ F_ini1
    del A_ini1, F_ini1

    A_ini2 = p_ini * ini_residual(W, b, pt_ini)
    F_ini2 = torch.zeros((pt_ini.shape[0], 1), dtype=dtype, device=device)
    ATA += A_ini2.T @ A_ini2
    ATF += A_ini2.T @ F_ini2
    del A_ini2, F_ini2

    ATA += ridge * torch.eye(M, dtype=dtype, device=device)
    Ur = torch.linalg.solve(ATA, ATF)
    return Ur

@torch.no_grad()
def predict_in_batches(x_test, t_test, W, b, Ur):
    Nx = x_test.shape[0]
    Nt = t_test.shape[0]

    u_pred_np = np.zeros((Nx, Nt), dtype=np.float64)

    W_x = W[0:1, :]
    W_t = W[1:2, :]
    X_part = x_test.unsqueeze(1) @ W_x

    for j0 in range(0, Nt, predict_chunk_size):
        j1 = min(j0 + predict_chunk_size, Nt)
        t_chunk = t_test[j0:j1]

        T_part = t_chunk.unsqueeze(1) @ W_t
        Z_chunk = X_part[:, None, :] + T_part[None, :, :] + b[None, :, :]
        phi_chunk = atv(Z_chunk)

        phi_2d = phi_chunk.reshape(-1, phi_chunk.shape[-1])
        u_chunk = (phi_2d @ Ur).reshape(Nx, j1 - j0)
        u_pred_np[:, j0:j1] = u_chunk.detach().cpu().numpy()

        del t_chunk, T_part, Z_chunk, phi_chunk, phi_2d, u_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return u_pred_np

# ============================================================
# Original single run for beta = 0.18 (to generate Figs 1-4)
# ============================================================
print("=" * 70)
print("Running fine-grid FDTD reference...")
print("=" * 70)
x_ref_full, t_ref_full, c_ref, q_ref, u_ref_save, probe_ref = run_fdtd_case(dx_ref, "Reference")

print("=" * 70)
print("Running coarse-grid FDTD baseline...")
print("=" * 70)
x_fdtd_full, t_fdtd_full, c_fdtd, q_fdtd, u_fdtd_save, probe_fdtd = run_fdtd_case(dx_fdtd, "Coarse FDTD")

u_ref_xt = u_ref_save.T
u_fdtd_xt = u_fdtd_save.T
fdtd_error_xt = u_fdtd_xt - u_ref_xt
rel_err_fdtd = relative_l2(u_fdtd_xt, u_ref_xt)

print(f"Coarse FDTD relative L2 error = {rel_err_fdtd:.2%}")

# RFNN training for beta=0.18
start_time = time.time()

W1 = torch.tensor(
    np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)),
    dtype=dtype,
    device=device
)
W2 = torch.tensor(
    np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)),
    dtype=dtype,
    device=device
)
W = torch.cat((W1, W2), dim=0)

b1 = torch.tensor(
    np.random.uniform(0.0, L, (1, num_neuron)),
    dtype=dtype,
    device=device
)
b2 = torch.tensor(
    np.random.uniform(0.0, T, (1, num_neuron)),
    dtype=dtype,
    device=device
)
b_shift = torch.cat((b1, b2), dim=0)
b = -torch.sum(b_shift * W, dim=0, keepdim=True)

pti, ptb, pt_ini = generate_points()
Ur = solve_blockwise_normal_equation(W, b, pti, ptb, pt_ini, ridge=1e-6)

x_test = torch.tensor(x_eval, device=device, dtype=dtype)
t_test = torch.tensor(t_eval, device=device, dtype=dtype)
u_rfnn_xt = predict_in_batches(x_test, t_test, W, b, Ur)

end_time = time.time()
print(f"RFNN total time = {end_time - start_time:.3f}s")

rfnn_error_xt = u_rfnn_xt - u_ref_xt
rel_err_rfnn = relative_l2(u_rfnn_xt, u_ref_xt)
print(f"RFNN relative L2 error = {rel_err_rfnn:.2%}")

# ============================================================
# Figure 1: medium profile
# ============================================================
x_medium = np.linspace(0.0, L, 1200)
c_medium = c_profile_np(x_medium)

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.plot(x_medium, c_medium, color=paper_colors["rfnn"], linewidth=2.6)
if "x1" in globals() and "x2" in globals():
    ax.axvline(x1, color="k", linestyle="--", alpha=0.30, linewidth=1.2)
    ax.axvline(x2, color="k", linestyle="--", alpha=0.30, linewidth=1.2)
ax.axvline(x_src, color=paper_colors["fdtd"], linestyle=":", alpha=0.9, linewidth=1.8)
ax.set_xlabel("Position $x$", fontweight="bold")
ax.set_ylabel("Wave speed $c(x)$", fontweight="bold")
ax.set_title("Nonuniform medium profile", fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.35)
ax.text(
    0.03, 0.93,
    "Dashed lines: nonuniform region\nDotted line: source location",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="0.8")
)
fig.tight_layout()
save_fig(fig, "fig1_medium_profile")
plt.show()

# ============================================================
# Figure 2: solution space-time maps
# ============================================================
vmax_sol = np.max(np.abs(np.concatenate([
    u_ref_save.ravel(),
    u_fdtd_save.ravel(),
    u_rfnn_xt.T.ravel()
])))
vmin_sol = -vmax_sol

fig = plt.figure(figsize=(18.5, 5.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.28)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])

im = axs[0].pcolormesh(x_eval, t_eval, u_ref_save, shading="auto", cmap="RdBu", vmin=vmin_sol, vmax=vmax_sol)
axs[0].set_title("(a) Fine-grid reference", fontweight="bold", pad=12)
axs[0].set_xlabel("Position x")
axs[0].set_ylabel("Time t")
axs[0].axvline(x_src, color="k", linestyle="--", alpha=0.35)

axs[1].pcolormesh(x_eval, t_eval, u_fdtd_save, shading="auto", cmap="RdBu", vmin=vmin_sol, vmax=vmax_sol)
axs[1].set_title(f"(b) Coarse FDTD\nRel. L2 = {rel_err_fdtd:.2%}", fontweight="bold", pad=12, linespacing=1.35)
axs[1].set_xlabel("Position x")
axs[1].set_ylabel("")
axs[1].tick_params(axis='y', labelleft=False)
axs[1].axvline(x_src, color="k", linestyle="--", alpha=0.35)

axs[2].pcolormesh(x_eval, t_eval, u_rfnn_xt.T, shading="auto", cmap="RdBu", vmin=vmin_sol, vmax=vmax_sol)
axs[2].set_title(f"(c) RFNN\nRel. L2 = {rel_err_rfnn:.2%}", fontweight="bold", pad=12, linespacing=1.35)
axs[2].set_xlabel("Position x")
axs[2].set_ylabel("")
axs[2].tick_params(axis='y', labelleft=False)
axs[2].axvline(x_src, color="k", linestyle="--", alpha=0.35)

cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_title("E(V/m)", fontsize=14, fontweight="bold", pad=8)

fig.suptitle("Space-time field distributions in the nonuniform medium", fontweight="bold", y=1.03)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.95)
save_fig(fig, "fig2_solution_maps")
plt.show()

# ============================================================
# Figure 3: error maps
# ============================================================
vmax_err = np.max(np.abs(np.concatenate([
    fdtd_error_xt.ravel(),
    rfnn_error_xt.ravel()
])))
vmin_err = -vmax_err

fig = plt.figure(figsize=(12.8, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])

im = ax1.pcolormesh(x_eval, t_eval, fdtd_error_xt.T, shading="auto", cmap="RdBu", vmin=vmin_err, vmax=vmax_err)
ax1.set_title(f"(a) Coarse FDTD error\nRel. L2 = {rel_err_fdtd:.2%}", fontweight="bold", pad=12, linespacing=1.35)
ax1.set_xlabel("Position x")
ax1.set_ylabel("Time t")
ax1.axvline(x_src, color="k", linestyle="--", alpha=0.35)

ax2.pcolormesh(x_eval, t_eval, rfnn_error_xt.T, shading="auto", cmap="RdBu", vmin=vmin_err, vmax=vmax_err)
ax2.set_title(f"(b) RFNN error\nRel. L2 = {rel_err_rfnn:.2%}", fontweight="bold", pad=12, linespacing=1.35)
ax2.set_xlabel("Position x")
ax2.set_ylabel("")
ax2.tick_params(axis='y', labelleft=False)
ax2.axvline(x_src, color="k", linestyle="--", alpha=0.35)

cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_title("Error(V/m)", fontsize=14, fontweight="bold", pad=8)

fig.suptitle("Error fields with respect to the fine-grid reference", fontweight="bold", y=1.03)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.94)
save_fig(fig, "fig3_error_maps")
plt.show()

# ============================================================
# Figure 4: probe traces
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.6))

for k, xpos in enumerate(probe_positions):
    idx = np.argmin(np.abs(x_eval - xpos))

    axs[k].plot(t_eval, u_ref_xt[idx], label="Reference", color=paper_colors["ref"])
    axs[k].plot(t_eval, u_fdtd_xt[idx], "--", label="FDTD", color=paper_colors["fdtd"])
    axs[k].plot(t_eval, u_rfnn_xt[idx], ":", label="RFNN", color=paper_colors["rfnn"])

    axs[k].set_xlabel("Time t")
    axs[k].set_ylabel("Field")
    axs[k].set_title(f"x = {x_eval[idx]:.2f}", fontweight="bold", pad=10)
    axs[k].grid(True, linestyle="--", alpha=0.35)
    if k == 0:
        axs[k].legend()

fig.suptitle("Appendix: Probe-trace comparison in the nonuniform medium", fontweight="bold", y=1.04)
fig.tight_layout()
save_fig(fig, "fig4_probe_traces")
plt.show()

print("\n" + "=" * 70)
print("Summary:")
print(f"Coarse FDTD relative L2 error = {rel_err_fdtd:.2%}")
print(f"RFNN         relative L2 error = {rel_err_rfnn:.2%}")
print("=" * 70)

# ============================================================
# NEW: L2 error vs beta study
# ============================================================
print("\n" + "=" * 70)
print("Starting L2 error vs beta study...")
print("=" * 70)

def compute_errors_for_beta(beta_value):
    """For a given beta, compute FDTD and RFNN relative L2 errors."""
    global beta   # modify global beta used in c_profile_np etc.
    beta = beta_value
    # Re-seed for reproducibility in each run
    torch.manual_seed(initial_seed)
    np.random.seed(initial_seed)

    print(f"\n--- beta = {beta:.2f} ---")

    # Fine-grid reference
    _, _, _, _, u_ref_save, _ = run_fdtd_case(dx_ref, "Reference")

    # Coarse FDTD
    _, _, _, _, u_fdtd_save, _ = run_fdtd_case(dx_fdtd, "Coarse FDTD")

    u_ref_xt = u_ref_save.T
    u_fdtd_xt = u_fdtd_save.T
    rel_err_fdtd = relative_l2(u_fdtd_xt, u_ref_xt)

    # RFNN training
    W1 = torch.tensor(
        np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)),
        dtype=dtype, device=device
    )
    W2 = torch.tensor(
        np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)),
        dtype=dtype, device=device
    )
    W = torch.cat((W1, W2), dim=0)

    b1 = torch.tensor(
        np.random.uniform(0.0, L, (1, num_neuron)),
        dtype=dtype, device=device
    )
    b2 = torch.tensor(
        np.random.uniform(0.0, T, (1, num_neuron)),
        dtype=dtype, device=device
    )
    b_shift = torch.cat((b1, b2), dim=0)
    b = -torch.sum(b_shift * W, dim=0, keepdim=True)

    pti, ptb, pt_ini = generate_points()
    Ur = solve_blockwise_normal_equation(W, b, pti, ptb, pt_ini, ridge=1e-6)

    x_test = torch.tensor(x_eval, device=device, dtype=dtype)
    t_test = torch.tensor(t_eval, device=device, dtype=dtype)
    u_rfnn_xt = predict_in_batches(x_test, t_test, W, b, Ur)

    rel_err_rfnn = relative_l2(u_rfnn_xt, u_ref_xt)

    print(f"  FDTD rel. L2 error: {rel_err_fdtd:.4f}")
    print(f"  RFNN  rel. L2 error: {rel_err_rfnn:.4f}")

    return rel_err_fdtd, rel_err_rfnn

beta_values = [0.05, 0.10, 0.18, 0.25, 0.35]
fdtd_errors = []
rfnn_errors = []

for bv in beta_values:
    err_fdtd, err_rfnn = compute_errors_for_beta(bv)
    fdtd_errors.append(err_fdtd)
    rfnn_errors.append(err_rfnn)

# Plot error vs beta
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(beta_values, np.array(fdtd_errors)*100, 'o-', color=paper_colors["fdtd"],
        label="Coarse FDTD", markersize=8, linewidth=2.2)
ax.plot(beta_values, np.array(rfnn_errors)*100, 's-', color=paper_colors["rfnn"],
        label="RFNN", markersize=8, linewidth=2.2)

ax.set_xlabel(r"$\beta$ (medium contrast)", fontweight="bold")
ax.set_ylabel("Relative L2 error (%)", fontweight="bold")
ax.set_title("Error vs. medium contrast parameter", fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend()
ax.set_xticks(beta_values)

fig.tight_layout()
save_fig(fig, "fig5_error_vs_beta")
plt.show()

print("\n" + "=" * 70)
print("L2 error vs beta results:")
for b, ef, er in zip(beta_values, fdtd_errors, rfnn_errors):
    print(f"beta={b:.2f}: FDTD={ef:.2%}, RFNN={er:.2%}")
print("=" * 70)
print(f"All paper figures saved in: {output_dir}")
