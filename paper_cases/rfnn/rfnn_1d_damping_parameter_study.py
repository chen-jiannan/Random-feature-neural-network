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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

# ============================================================
# Global plotting style
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

paper_colors = {
    "rfnn": "#1f77b4",
    "fdtd": "#d62728",
    "gray": "#7f7f7f",
    "green": "#2ca02c",
    "purple": "#9467bd",
    "orange": "#ff7f0e",
}

def save_fig(fig, out_dir, name):
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.35)

# ============================================================
# Common settings
# ============================================================
output_dir = "sigma_parameter_study_gpu_double"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Physical parameters
# -----------------------------
c = 1.0
L = 5.0
T = 5.0
x_src = 2.5
freq = 1.0
omega = 2.0 * np.pi * freq
source_sigma = 0.1

probe_positions = [0.1, 0.2, 4.8, 4.9]

# -----------------------------
# Absorbing layer settings
# -----------------------------
USE_PML = True
pml_width = 1.5

# -----------------------------
# FDTD settings (fixed)
# -----------------------------
fdtd_dx = 1e-4
fdtd_dt = fdtd_dx / c

# -----------------------------
# RFNN settings
# -----------------------------
initial_seed = 5
num_neuron = 3000
r_neuron = [5, 5]
N_pde = 30000
N_bnd = 20000
N_ini = 10000
p_pde = 1.0
p_bnd = 10.0
p_ini_u = 10.0
p_ini_v = 10.0
solve_block_size = 5000
predict_chunk_size_t = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

print(f"Using device: {device}")
print(f"Using dtype  : {dtype}")

# -----------------------------
# Parameter study plan
# -----------------------------
sigma_max_list = [5.0, 10.0, 20.0, 40.0, 80.0]
sigma_order_list = [2, 3, 4, 5]
fixed_sigma_max_for_order = 20.0
fixed_sigma_order_for_max = 3.0

# ============================================================
# Shared helpers
# ============================================================
def get_domain_bounds():
    return -pml_width, L + pml_width

x_left, x_right = get_domain_bounds()

def interpolate_trace_along_x(x_grid, u_xt, x_query):
    if x_query <= x_grid[0]:
        return u_xt[0, :].copy()
    if x_query >= x_grid[-1]:
        return u_xt[-1, :].copy()

    i1 = np.searchsorted(x_grid, x_query)
    i0 = i1 - 1
    x0 = x_grid[i0]
    x1 = x_grid[i1]
    alpha = (x_query - x0) / (x1 - x0)
    return (1.0 - alpha) * u_xt[i0, :] + alpha * u_xt[i1, :]

def interpolate_trace_in_time(t_old, trace_old, t_new):
    return np.interp(t_new, t_old, trace_old)

def compute_probe_metrics(x_num, t_num, u_num_xt, x_ref, t_ref, u_ref_xt, probe_positions, eps=1e-15):
    results = []

    for xpos in probe_positions:
        trace_num = interpolate_trace_along_x(x_num, u_num_xt, xpos)
        trace_ref_raw = interpolate_trace_along_x(x_ref, u_ref_xt, xpos)

        if len(t_ref) != len(t_num) or not np.allclose(t_ref, t_num):
            trace_ref = interpolate_trace_in_time(t_ref, trace_ref_raw, t_num)
        else:
            trace_ref = trace_ref_raw

        diff_trace = trace_num - trace_ref
        numerator = np.max(np.abs(diff_trace))
        denominator = np.max(np.abs(trace_ref))

        if denominator < eps:
            R = np.nan
            R_dB = np.nan
        else:
            R = numerator / denominator
            R_dB = 20.0 * np.log10(max(R, eps))

        results.append({
            "x": xpos,
            "R_linear": R,
            "R_dB": R_dB,
            "trace_num": trace_num,
            "trace_ref": trace_ref,
            "trace_diff": diff_trace
        })

    return results

def compute_worst_metric_in_region(x_num, t_num, u_num_xt, x_ref, t_ref, u_ref_xt,
                                   x_region_min, x_region_max, n_probe=50, eps=1e-15):
    probe_x = np.linspace(x_region_min, x_region_max, n_probe)
    results = compute_probe_metrics(
        x_num, t_num, u_num_xt,
        x_ref, t_ref, u_ref_xt,
        probe_x, eps=eps
    )
    valid = [r for r in results if np.isfinite(r["R_linear"])]
    if len(valid) == 0:
        return None
    return max(valid, key=lambda d: d["R_linear"])

def summarize_case(method_name, sigma_max, sigma_order, x_num, t_num, u_num_xt, x_ref, t_ref, u_ref_xt):
    rel_l2 = np.linalg.norm(u_num_xt - u_ref_xt) / max(np.linalg.norm(u_ref_xt), 1e-15)

    probe_results = compute_probe_metrics(
        x_num, t_num, u_num_xt,
        x_ref, t_ref, u_ref_xt,
        probe_positions
    )

    left_worst = compute_worst_metric_in_region(
        x_num, t_num, u_num_xt,
        x_ref, t_ref, u_ref_xt,
        0.0, 0.5, n_probe=50
    )
    right_worst = compute_worst_metric_in_region(
        x_num, t_num, u_num_xt,
        x_ref, t_ref, u_ref_xt,
        4.5, 5.0, n_probe=50
    )

    mean_probe_db = np.nanmean([item["R_dB"] for item in probe_results])

    row = {
        "method": method_name,
        "sigma_max": sigma_max,
        "sigma_order": sigma_order,
        "rel_l2": rel_l2,
        "probe_x0.1_dB": probe_results[0]["R_dB"],
        "probe_x0.2_dB": probe_results[1]["R_dB"],
        "probe_x4.8_dB": probe_results[2]["R_dB"],
        "probe_x4.9_dB": probe_results[3]["R_dB"],
        "mean_probe_dB": mean_probe_db,
        "left_worst_dB": np.nan if left_worst is None else left_worst["R_dB"],
        "right_worst_dB": np.nan if right_worst is None else right_worst["R_dB"],
    }
    return row, probe_results

# ============================================================
# Reference solution
# ============================================================
reference_file = "1d_fdtd_ref_5e-5.npy"
u_ref_xt = np.load(reference_file).T.astype(np.float64)

x_ref = np.linspace(0.0, L, u_ref_xt.shape[0])
t_ref = np.linspace(0.0, T, u_ref_xt.shape[1])

print(f"Loaded reference file: {reference_file}")
print(f"Reference shape = {u_ref_xt.shape}")

# ============================================================
# GPU FDTD damping solver (double precision)
# ============================================================
def run_fdtd_case_gpu(sigma_max, sigma_order):
    dx = fdtd_dx
    dt = fdtd_dt

    n_pml = int(round(pml_width / dx))
    actual_pml_width = n_pml * dx

    nx_total = int(round(L / dx)) + 1
    nx = nx_total + 2 * n_pml
    nt = int(round(T / dt)) + 1

    x = np.arange(nx, dtype=np.float64) * dx - actual_pml_width
    x_main = np.linspace(0.0, L, nx_total, dtype=np.float64)
    t = np.linspace(0.0, T, nt, dtype=np.float64)

    save_t_idx = np.array([np.argmin(np.abs(t - tt)) for tt in t_ref], dtype=np.int64)
    save_x_idx_main_local = np.array([np.argmin(np.abs(x_main - xx)) for xx in x_ref], dtype=np.int64)
    save_x_idx_main = save_x_idx_main_local + n_pml

    probe_positions_np = np.array(probe_positions, dtype=np.float64)
    probe_idx_main_local = np.array([np.argmin(np.abs(x_main - xp)) for xp in probe_positions_np], dtype=np.int64)
    probe_idx_main = probe_idx_main_local + n_pml

    x_t = torch.tensor(x, device=device, dtype=dtype)
    gaussian_source_weights = torch.exp(-((x_t - x_src) ** 2) / (2.0 * source_sigma ** 2))

    sigma = torch.zeros(nx, device=device, dtype=dtype)
    left_mask = x_t < 0.0
    right_mask = x_t > L
    sigma[left_mask] = sigma_max * ((0.0 - x_t[left_mask]) / pml_width) ** sigma_order
    sigma[right_mask] = sigma_max * ((x_t[right_mask] - L) / pml_width) ** sigma_order

    denom = 1.0 + sigma * dt / 2.0
    c1 = 2.0 / denom
    c2 = (1.0 - sigma * dt / 2.0) / denom
    c3 = ((c * dt / dx) ** 2) / denom
    c_src = (dt ** 2) / denom

    save_t_idx_t = torch.tensor(save_t_idx, device=device, dtype=torch.long)
    save_x_idx_main_t = torch.tensor(save_x_idx_main, device=device, dtype=torch.long)
    probe_idx_main_t = torch.tensor(probe_idx_main, device=device, dtype=torch.long)

    Ex = torch.zeros(nx, device=device, dtype=dtype)
    Ex_prev = torch.zeros(nx, device=device, dtype=dtype)
    Ex_next = torch.zeros(nx, device=device, dtype=dtype)

    Ex_save_main = np.zeros((len(t_ref), len(x_ref)), dtype=np.float64)
    probe_traces = np.zeros((len(probe_positions), nt), dtype=np.float64)

    save_counter = 0
    two_pi_freq_dt = 2.0 * np.pi * freq * dt

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for n in range(nt):
        Ex_next[1:-1] = (
            c1[1:-1] * Ex[1:-1]
            - c2[1:-1] * Ex_prev[1:-1]
            + c3[1:-1] * (Ex[2:] - 2.0 * Ex[1:-1] + Ex[:-2])
        )

        amplitude = 0.5 if n < 1 else 1.0
        source_phase = amplitude * np.cos(two_pi_freq_dt * n)
        Ex_next += gaussian_source_weights * source_phase * c_src

        Ex_next[0] = 0.0
        Ex_next[-1] = 0.0

        probe_traces[:, n] = Ex.index_select(0, probe_idx_main_t).detach().cpu().numpy()

        if save_counter < len(t_ref) and n == save_t_idx[save_counter]:
            Ex_save_main[save_counter] = Ex.index_select(0, save_x_idx_main_t).detach().cpu().numpy()
            save_counter += 1

        Ex_prev, Ex, Ex_next = Ex, Ex_next, Ex_prev
        Ex_next.zero_()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(f"FDTD case done: sigma_max={sigma_max}, sigma_order={sigma_order}, time={elapsed:.3f}s")

    u_fdtd_main_xt = Ex_save_main.T  # shape = (Nx_ref, Nt_ref)
    return x_ref.copy(), t_ref.copy(), u_fdtd_main_xt

# ============================================================
# RFNN damping solver
# ============================================================
def sigma_x_torch(X, sigma_max, sigma_order):
    X = X.to(dtype)
    sig = torch.zeros_like(X, dtype=dtype, device=X.device)

    if not USE_PML:
        return sig

    left_mask = X < 0.0
    if left_mask.any():
        sig[left_mask] = sigma_max * ((0.0 - X[left_mask]) / pml_width) ** sigma_order

    right_mask = X > L
    if right_mask.any():
        sig[right_mask] = sigma_max * ((X[right_mask] - L) / pml_width) ** sigma_order

    return sig

def generate_points_rfnn():
    N_uniform = N_pde
    N_gaussian = N_pde - N_uniform

    xi_uniform = x_left + (x_right - x_left) * torch.rand(
        N_uniform, 1, device=device, dtype=dtype
    )
    ti_uniform = T * torch.rand(
        N_uniform, 1, device=device, dtype=dtype
    )

    xi_gaussian = torch.normal(
        mean=x_src,
        std=0.1,
        size=(N_gaussian, 1),
        device=device
    ).to(dtype).clamp(x_left, x_right)

    ti_gaussian = T * torch.rand(
        N_gaussian, 1, device=device, dtype=dtype
    )

    xi = torch.cat((xi_uniform, xi_gaussian), dim=0)
    ti = torch.cat((ti_uniform, ti_gaussian), dim=0)
    pti = torch.cat((xi, ti), dim=1).to(dtype)

    tb = T * torch.rand(N_bnd, 1, device=device, dtype=dtype)
    ptb_left = torch.cat((torch.full_like(tb, x_left), tb), dim=1)
    ptb_right = torch.cat((torch.full_like(tb, x_right), tb), dim=1)
    ptb = torch.cat((ptb_left, ptb_right), dim=0).to(dtype)

    xi_ini = x_left + (x_right - x_left) * torch.rand(
        N_ini, 1, device=device, dtype=dtype
    )
    ti_ini = torch.zeros(N_ini, 1, device=device, dtype=dtype)
    pt_ini = torch.cat((xi_ini, ti_ini), dim=1).to(dtype)

    return pti, ptb, pt_ini

def atv(x):
    return torch.tanh(x)

def diff1_atv(x):
    return 1.0 - torch.tanh(x) ** 2

def diff2_atv(x):
    th = torch.tanh(x)
    return 2.0 * th * (th**2 - 1.0)

def feature_value(W, b, points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]
    Z = X @ W_x + Tt @ W_t + b
    return atv(Z)

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

def source_term_rfnn(points):
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    return torch.cos(2.0 * np.pi * freq * Tt) * torch.exp(-((X - x_src) ** 2) / 0.02)

def u0(x):
    return torch.zeros_like(x)

def v0(x):
    return torch.zeros_like(x)

def wave_equation_residual_rfnn(W, b, points, sigma_max, sigma_order):
    phi_tt = feature_tt(W, b, points)
    phi_xx = feature_xx(W, b, points)
    X = points[:, 0:1]
    sig = sigma_x_torch(X, sigma_max, sigma_order)
    phi_t = feature_t(W, b, points)
    return phi_tt + sig * phi_t - (c ** 2) * phi_xx

@torch.no_grad()
def solve_blockwise_normal_equation_rfnn(W, b, pti, ptb, pt_ini, sigma_max, sigma_order, ridge=1e-6):
    M = W.shape[1]
    ATA = torch.zeros((M, M), dtype=dtype, device=device)
    ATF = torch.zeros((M, 1), dtype=dtype, device=device)

    n_pde_total = pti.shape[0]

    for i0 in range(0, n_pde_total, solve_block_size):
        i1 = min(i0 + solve_block_size, n_pde_total)
        pti_block = pti[i0:i1]

        A_pde_block = p_pde * wave_equation_residual_rfnn(W, b, pti_block, sigma_max, sigma_order)
        F_pde_block = p_pde * source_term_rfnn(pti_block).to(dtype)

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    A_ini_u = p_ini_u * feature_value(W, b, pt_ini)
    F_ini_u = p_ini_u * u0(pt_ini[:, 0:1]).to(dtype)
    ATA += A_ini_u.T @ A_ini_u
    ATF += A_ini_u.T @ F_ini_u
    del A_ini_u, F_ini_u
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    A_ini_v = p_ini_v * feature_t(W, b, pt_ini)
    F_ini_v = p_ini_v * v0(pt_ini[:, 0:1]).to(dtype)
    ATA += A_ini_v.T @ A_ini_v
    ATF += A_ini_v.T @ F_ini_v
    del A_ini_v, F_ini_v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ATA += ridge * torch.eye(M, dtype=dtype, device=device)
    Ur = torch.linalg.solve(ATA, ATF)
    return Ur

@torch.no_grad()
def predict_main_grid_in_batches_rfnn(x_test, t_test, W, b, Ur):
    Nx = x_test.shape[0]
    Nt = t_test.shape[0]
    u_pred_np = np.zeros((Nx, Nt), dtype=np.float64)

    W_x = W[0:1, :]
    W_t = W[1:2, :]
    X_part = x_test.unsqueeze(1) @ W_x

    for j0 in range(0, Nt, predict_chunk_size_t):
        j1 = min(j0 + predict_chunk_size_t, Nt)
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

def run_rfnn_case(sigma_max, sigma_order):
    torch.manual_seed(initial_seed)
    np.random.seed(initial_seed)

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
        np.random.uniform(x_left, x_right, (1, num_neuron)),
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

    pti, ptb, pt_ini = generate_points_rfnn()

    start = time.perf_counter()
    Ur = solve_blockwise_normal_equation_rfnn(
        W=W, b=b,
        pti=pti, ptb=ptb, pt_ini=pt_ini,
        sigma_max=sigma_max, sigma_order=sigma_order,
        ridge=1e-6
    )

    x_test_main = torch.tensor(x_ref, device=device, dtype=dtype)
    t_test_main = torch.tensor(t_ref, device=device, dtype=dtype)

    u_pred_main = predict_main_grid_in_batches_rfnn(
        x_test=x_test_main,
        t_test=t_test_main,
        W=W,
        b=b,
        Ur=Ur
    )
    elapsed = time.perf_counter() - start

    print(f"RFNN case done: sigma_max={sigma_max}, sigma_order={sigma_order}, time={elapsed:.3f}s")
    return x_ref.copy(), t_ref.copy(), u_pred_main

# ============================================================
# Parameter study
# ============================================================
all_rows = []

# -----------------------------
# Study 1: sigma_max
# -----------------------------
print("\n" + "=" * 80)
print("Study 1: scan sigma_max with sigma_order fixed")
print("=" * 80)

for smax in sigma_max_list:
    x_fdtd, t_fdtd, u_fdtd = run_fdtd_case_gpu(sigma_max=smax, sigma_order=fixed_sigma_order_for_max)
    row_fdtd, _ = summarize_case(
        method_name="FDTD",
        sigma_max=smax,
        sigma_order=fixed_sigma_order_for_max,
        x_num=x_fdtd, t_num=t_fdtd, u_num_xt=u_fdtd,
        x_ref=x_ref, t_ref=t_ref, u_ref_xt=u_ref_xt
    )
    row_fdtd["study"] = "sigma_max"
    all_rows.append(row_fdtd)

    x_rfnn, t_rfnn, u_rfnn = run_rfnn_case(sigma_max=smax, sigma_order=fixed_sigma_order_for_max)
    row_rfnn, _ = summarize_case(
        method_name="RFNN",
        sigma_max=smax,
        sigma_order=fixed_sigma_order_for_max,
        x_num=x_rfnn, t_num=t_rfnn, u_num_xt=u_rfnn,
        x_ref=x_ref, t_ref=t_ref, u_ref_xt=u_ref_xt
    )
    row_rfnn["study"] = "sigma_max"
    all_rows.append(row_rfnn)

# -----------------------------
# Study 2: sigma_order
# -----------------------------
print("\n" + "=" * 80)
print("Study 2: scan sigma_order with sigma_max fixed")
print("=" * 80)

for sord in sigma_order_list:
    x_fdtd, t_fdtd, u_fdtd = run_fdtd_case_gpu(sigma_max=fixed_sigma_max_for_order, sigma_order=sord)
    row_fdtd, _ = summarize_case(
        method_name="FDTD",
        sigma_max=fixed_sigma_max_for_order,
        sigma_order=sord,
        x_num=x_fdtd, t_num=t_fdtd, u_num_xt=u_fdtd,
        x_ref=x_ref, t_ref=t_ref, u_ref_xt=u_ref_xt
    )
    row_fdtd["study"] = "sigma_order"
    all_rows.append(row_fdtd)

    x_rfnn, t_rfnn, u_rfnn = run_rfnn_case(sigma_max=fixed_sigma_max_for_order, sigma_order=sord)
    row_rfnn, _ = summarize_case(
        method_name="RFNN",
        sigma_max=fixed_sigma_max_for_order,
        sigma_order=sord,
        x_num=x_rfnn, t_num=t_rfnn, u_num_xt=u_rfnn,
        x_ref=x_ref, t_ref=t_ref, u_ref_xt=u_ref_xt
    )
    row_rfnn["study"] = "sigma_order"
    all_rows.append(row_rfnn)

# ============================================================
# Unified paper-style plotting / table export
# ============================================================
df = pd.DataFrame(all_rows)
df = df.sort_values(by=["study", "sigma_max", "sigma_order", "method"]).reset_index(drop=True)

csv_path = os.path.join(output_dir, "sigma_parameter_study_table.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved table: {csv_path}")

summary_rows = []

df_max = df[df["study"] == "sigma_max"].copy()
for smax in sigma_max_list:
    sub = df_max[df_max["sigma_max"] == smax]
    row = {"Study": "sigma_max", "Parameter": smax}

    for method in ["RFNN", "FDTD"]:
        sm = sub[sub["method"] == method].iloc[0]
        row[f"{method} RelL2"] = sm["rel_l2"]
        row[f"{method} MeanProbe_dB"] = sm["mean_probe_dB"]
        row[f"{method} LeftWorst_dB"] = sm["left_worst_dB"]
        row[f"{method} RightWorst_dB"] = sm["right_worst_dB"]

    summary_rows.append(row)

df_order = df[df["study"] == "sigma_order"].copy()
for sord in sigma_order_list:
    sub = df_order[df_order["sigma_order"] == sord]
    row = {"Study": "sigma_order", "Parameter": sord}

    for method in ["RFNN", "FDTD"]:
        sm = sub[sub["method"] == method].iloc[0]
        row[f"{method} RelL2"] = sm["rel_l2"]
        row[f"{method} MeanProbe_dB"] = sm["mean_probe_dB"]
        row[f"{method} LeftWorst_dB"] = sm["left_worst_dB"]
        row[f"{method} RightWorst_dB"] = sm["right_worst_dB"]

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(output_dir, "sigma_parameter_study_summary_table.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved summary table: {summary_csv_path}")

print("\n===== Summary Table =====")
print(summary_df.to_string(index=False))

# ============================================================
# Figure 1: sigma_max scan
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.4))

for method, color, marker in [("RFNN", paper_colors["rfnn"], "o"), ("FDTD", paper_colors["fdtd"], "s")]:
    sub = df_max[df_max["method"] == method].sort_values("sigma_max")

    axs[0].plot(
        sub["sigma_max"], sub["rel_l2"],
        marker=marker, color=color, linewidth=2.4, markersize=7, label=method
    )
    axs[1].plot(
        sub["sigma_max"], sub["mean_probe_dB"],
        marker=marker, color=color, linewidth=2.4, markersize=7, label=method
    )

axs[0].set_xlabel(r"$\sigma_{\max}$", fontweight="bold")
axs[0].set_ylabel("Relative L2 Error", fontweight="bold")
axs[0].set_title(r"(a) Global Error vs. $\sigma_{\max}$", fontweight="bold", pad=12)
style_axis(axs[0], grid_axis="both")
axs[0].legend(frameon=True)

axs[1].set_xlabel(r"$\sigma_{\max}$", fontweight="bold")
axs[1].set_ylabel(r"Mean Probe $R_{\mathrm{dB}}$", fontweight="bold")
axs[1].set_title(r"(b) Probe Deviation vs. $\sigma_{\max}$", fontweight="bold", pad=12)
style_axis(axs[1], grid_axis="both")
axs[1].legend(frameon=True)

fig.suptitle(
    rf"Influence of $\sigma_{{\max}}$ with fixed $\sigma_{{\mathrm{{order}}}}={fixed_sigma_order_for_max}$",
    fontweight="bold",
    y=1.02
)

fig.tight_layout()
save_fig(fig, output_dir, "fig_sigma_max_study")
plt.show()

# ============================================================
# Figure 2: sigma_order scan
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.4))

for method, color, marker in [("RFNN", paper_colors["rfnn"], "o"), ("FDTD", paper_colors["fdtd"], "s")]:
    sub = df_order[df_order["method"] == method].sort_values("sigma_order")

    axs[0].plot(
        sub["sigma_order"], sub["rel_l2"],
        marker=marker, color=color, linewidth=2.4, markersize=7, label=method
    )
    axs[1].plot(
        sub["sigma_order"], sub["mean_probe_dB"],
        marker=marker, color=color, linewidth=2.4, markersize=7, label=method
    )

axs[0].set_xlabel(r"$\sigma_{\mathrm{order}}$", fontweight="bold")
axs[0].set_ylabel("Relative L2 Error", fontweight="bold")
axs[0].set_title(r"(a) Global Error vs. $\sigma_{\mathrm{order}}$", fontweight="bold", pad=12)
style_axis(axs[0], grid_axis="both")
axs[0].legend(frameon=True)

axs[1].set_xlabel(r"$\sigma_{\mathrm{order}}$", fontweight="bold")
axs[1].set_ylabel(r"Mean Probe $R_{\mathrm{dB}}$", fontweight="bold")
axs[1].set_title(r"(b) Probe Deviation vs. $\sigma_{\mathrm{order}}$", fontweight="bold", pad=12)
style_axis(axs[1], grid_axis="both")
axs[1].legend(frameon=True)

fig.suptitle(
    rf"Influence of $\sigma_{{\mathrm{{order}}}}$ with fixed $\sigma_{{\max}}={fixed_sigma_max_for_order}$",
    fontweight="bold",
    y=1.02
)

fig.tight_layout()
save_fig(fig, output_dir, "fig_sigma_order_study")
plt.show()

print("\nAll done.")
print(f"Results saved in: {output_dir}")
