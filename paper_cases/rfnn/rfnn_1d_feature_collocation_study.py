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
import torch
import matplotlib.pyplot as plt

# =========================================================
# Basic Configuration
# =========================================================
initial_seed = 42
torch.manual_seed(initial_seed)
np.random.seed(initial_seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

c = 1.0
L = 5.0
T = 5.0
r_neuron = [5, 5]

# Penalty weights
p_pde = 1.0
p_bnd = 10.0
p_ini = 10.0

# Source parameters
sigma = 0.1
omega = 2 * np.pi

# Test grid
N_test_t = 101
N_test_x = 101

# Reference file
REFERENCE_FILE = DATA_DIR / '1d_fdtd_two_sources_cos_1e-4.npy'


# =========================================================
# Global Plot Style
# =========================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
})


# =========================================================
# Activation and Derivatives
# =========================================================
def atv(x):
    return torch.tanh(x)

def diff1_atv(x):
    return 1.0 - torch.tanh(x) ** 2

def diff2_atv(x):
    return 2.0 * torch.tanh(x) * (torch.tanh(x) ** 2 - 1.0)


# =========================================================
# Data Generation
# =========================================================
def generate_points(N_pde=30000, N_bnd=20000, N_ini=10000, device=device):
    # PDE points
    xi = torch.rand(N_pde, 1, device=device) * L
    ti = torch.rand(N_pde, 1, device=device) * T
    pti = torch.cat((ti, xi), dim=1)

    # Boundary points
    tb = torch.rand(N_bnd, 1, device=device) * T
    ptb_0 = torch.cat((tb, torch.zeros_like(tb)), dim=1)
    ptb_L = torch.cat((tb, torch.full_like(tb, L)), dim=1)
    ptb = torch.cat((ptb_0, ptb_L), dim=0)

    # Initial points
    xi_ini = torch.rand(N_ini, 1, device=device) * L
    ti_ini = torch.zeros(N_ini, 1, device=device)
    pt_ini = torch.cat((ti_ini, xi_ini), dim=1)

    return pti, ptb, pt_ini


# =========================================================
# Physics Residuals
# =========================================================
def wave_equation_residual(W, b, points, c):
    T_col = points[:, 0:1]
    X_col = points[:, 1:2]

    W_t = W[0:1, :]
    W_x = W[1:2, :]

    Z = T_col @ W_t + X_col @ W_x + b
    phi_tt = (W_t ** 2) * diff2_atv(Z)
    phi_xx = (W_x ** 2) * diff2_atv(Z)

    return phi_tt - (c ** 2) * phi_xx


def ini_residual(W, b, points):
    T_col = points[:, 0:1]
    X_col = points[:, 1:2]

    W_t = W[0:1, :]
    W_x = W[1:2, :]

    Z = T_col @ W_t + X_col @ W_x + b
    phi_t = W_t * diff1_atv(Z)

    return phi_t


# =========================================================
# RFNN Initialization
# =========================================================
def initialize_random_features(num_neuron, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    W1 = torch.tensor(
        np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)),
        dtype=torch.float32,
        device=device
    )
    W2 = torch.tensor(
        np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)),
        dtype=torch.float32,
        device=device
    )
    W = torch.cat((W1, W2), dim=0)

    b1 = torch.tensor(
        np.random.uniform(0.0, L, (1, num_neuron)),
        dtype=torch.float32,
        device=device
    )
    b2 = torch.tensor(
        np.random.uniform(0.0, T, (1, num_neuron)),
        dtype=torch.float32,
        device=device
    )
    b_raw = torch.cat((b1, b2), dim=0)

    # Bias correction
    b = -torch.sum(b_raw * W, dim=0)

    return W, b


# =========================================================
# Source Term
# =========================================================
def build_rhs_pde(pti):
    source_term1 = 2.0 * torch.cos(omega * pti[:, 0:1]) * torch.exp(
        -((pti[:, 1:2] - 1.5) ** 2) / (2.0 * sigma ** 2)
    )
    source_term2 = 5.0 * torch.cos(2.0 * omega * pti[:, 0:1]) * torch.exp(
        -((pti[:, 1:2] - 4.0) ** 2) / (2.0 * sigma ** 2)
    )
    return source_term1 + source_term2


# =========================================================
# Reference Solution
# =========================================================
def load_reference_solution():
    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"Reference file '{REFERENCE_FILE}' not found. "
            "Please place it in the current working directory."
        )

    u_ref = np.load(REFERENCE_FILE)

    # Downsample to 101x101
    u_ref = u_ref[0::5, 0::5]

    if u_ref.shape != (N_test_t, N_test_x):
        raise ValueError(
            f"Reference shape after downsampling is {u_ref.shape}, "
            f"but expected {(N_test_t, N_test_x)}."
        )

    return u_ref


# =========================================================
# Prediction Grid
# =========================================================
def make_test_grid():
    t_test = torch.linspace(0, T, N_test_t, device=device)
    x_test = torch.linspace(0, L, N_test_x, device=device)
    T_mesh, X_mesh = torch.meshgrid(t_test, x_test, indexing='ij')
    test_points = torch.cat(
        (T_mesh.reshape(-1, 1), X_mesh.reshape(-1, 1)),
        dim=1
    )
    return T_mesh, X_mesh, test_points


# =========================================================
# Single Experiment
# =========================================================
def run_single_experiment(
    num_neuron=5000,
    N_pde=30000,
    N_bnd=20000,
    N_ini=10000,
    seed=42,
    u_ref=None,
    test_points=None,
    T_mesh=None,
):
    # Initialize random features
    W, b = initialize_random_features(num_neuron=num_neuron, seed=seed)

    # Generate collocation points
    pti, ptb, pt_ini = generate_points(
        N_pde=N_pde,
        N_bnd=N_bnd,
        N_ini=N_ini,
        device=device
    )

    # =====================================================
    # Computing time:
    # matrix assembly + RHS construction + linear solve
    # =====================================================
    compute_start = time.time()

    A_pde = wave_equation_residual(W, b, pti, c)
    A_bnd = atv(ptb @ W + b)
    A_ini1 = atv(pt_ini @ W + b)
    A_ini2 = ini_residual(W, b, pt_ini)

    A = torch.cat(
        [p_pde * A_pde, p_bnd * A_bnd, p_ini * A_ini1, p_ini * A_ini2],
        dim=0
    )

    F_pde = build_rhs_pde(pti)
    F_bnd = torch.zeros((ptb.shape[0], 1), device=device)
    F_ini1 = torch.zeros((pt_ini.shape[0], 1), device=device)
    F_ini2 = torch.zeros((pt_ini.shape[0], 1), device=device)

    F_vec = torch.cat(
        [p_pde * F_pde, p_bnd * F_bnd, p_ini * F_ini1, p_ini * F_ini2],
        dim=0
    )

    Ur = torch.linalg.lstsq(A, F_vec).solution

    computing_time = time.time() - compute_start

    # Prediction only used to evaluate error
    Z_test = test_points @ W + b
    u_pred = (atv(Z_test) @ Ur).reshape(T_mesh.shape)
    u_pred_np = u_pred.detach().cpu().numpy()

    rel_err = np.linalg.norm(u_pred_np - u_ref) / np.linalg.norm(u_ref)

    return {
        "num_neuron": num_neuron,
        "N_pde": N_pde,
        "N_bnd": N_bnd,
        "N_ini": N_ini,
        "sampling_points": N_pde + N_bnd + N_ini,
        "computing_time": computing_time,
        "rel_err": rel_err,
    }


# =========================================================
# Study 1: Hidden Features vs Error / Computing Time
# =========================================================
def study_hidden_features(
    neuron_list,
    fixed_N_pde=30000,
    fixed_N_bnd=20000,
    fixed_N_ini=10000,
    seed=42
):
    u_ref = load_reference_solution()
    T_mesh, _, test_points = make_test_grid()

    results = []
    for m in neuron_list:
        print(f"\n[Hidden Feature Study] num_neuron = {m}")
        out = run_single_experiment(
            num_neuron=m,
            N_pde=fixed_N_pde,
            N_bnd=fixed_N_bnd,
            N_ini=fixed_N_ini,
            seed=seed,
            u_ref=u_ref,
            test_points=test_points,
            T_mesh=T_mesh,
        )
        print(
            f"  rel_err = {out['rel_err']:.4%}, "
            f"computing_time = {out['computing_time']:.3f}s"
        )
        results.append(out)

    return results


# =========================================================
# Study 2: Collocation Points vs Error
# =========================================================
def study_collocation_points(
    collocation_scales,
    base_N_pde=30000,
    base_N_bnd=20000,
    base_N_ini=10000,
    fixed_num_neuron=5000,
    seed=42
):
    u_ref = load_reference_solution()
    T_mesh, _, test_points = make_test_grid()

    results = []
    for scale in collocation_scales:
        N_pde = int(base_N_pde * scale)
        N_bnd = int(base_N_bnd * scale)
        N_ini = int(base_N_ini * scale)

        print(
            f"\n[Collocation Study] scale = {scale}, "
            f"N_pde = {N_pde}, N_bnd = {N_bnd}, N_ini = {N_ini}"
        )

        out = run_single_experiment(
            num_neuron=fixed_num_neuron,
            N_pde=N_pde,
            N_bnd=N_bnd,
            N_ini=N_ini,
            seed=seed,
            u_ref=u_ref,
            test_points=test_points,
            T_mesh=T_mesh,
        )
        print(
            f"  rel_err = {out['rel_err']:.4%}, "
            f"computing_time = {out['computing_time']:.3f}s"
        )
        results.append(out)

    return results


# =========================================================
# Plotting
# =========================================================
def plot_hidden_features_vs_error_time(
    results,
    save_path="hidden_features_vs_error_time.pdf"
):
    neuron_vals = [r["num_neuron"] for r in results]
    rel_err_vals = [r["rel_err"] * 100.0 for r in results]
    comp_time_vals = [r["computing_time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8.2, 5.8))

    # Left axis: Relative L2 Error
    line1 = ax1.plot(
        neuron_vals, rel_err_vals,
        color='tab:blue',
        linestyle='-',
        marker='o',
        markersize=9,
        linewidth=2.4,
        markeredgewidth=1.0,
        label='Relative L2 Error'
    )
    ax1.set_xlabel("Number of Hidden Features", fontweight='bold')
    ax1.set_ylabel("Relative L2 Error (%)", fontweight='bold')
    ax1.tick_params(axis='both', width=1.2)
    ax1.grid(alpha=0.28, linestyle='--')

    # Right axis: Computing Time
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        neuron_vals, comp_time_vals,
        color='tab:red',
        linestyle='--',
        marker='s',
        markersize=9,
        linewidth=2.4,
        markeredgewidth=1.0,
        label='Computing Time'
    )
    ax2.set_ylabel("Computing Time (s)", fontweight='bold')
    ax2.tick_params(axis='both', width=1.2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc="best", frameon=True)
    for txt in legend.get_texts():
        txt.set_fontweight('bold')

    plt.title("Hidden Features vs Error and Computing Time", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_collocation_vs_error_time(
    results,
    save_path="collocation_vs_error_time.pdf"
):
    sampling_points = [r["sampling_points"] for r in results]
    rel_err_vals = [r["rel_err"] * 100.0 for r in results]
    comp_time_vals = [r["computing_time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8.2, 5.8))

    line1 = ax1.plot(
        sampling_points, rel_err_vals,
        color='tab:blue',
        linestyle='-',
        marker='o',
        markersize=9,
        linewidth=2.4,
        markeredgewidth=1.0,
        label='Relative L2 Error'
    )
    ax1.set_xlabel("Number of Sampling Points", fontweight='bold')
    ax1.set_ylabel("Relative L2 Error (%)", fontweight='bold')
    ax1.tick_params(axis='both', width=1.2)
    ax1.grid(alpha=0.28, linestyle='--')

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        sampling_points, comp_time_vals,
        color='tab:red',
        linestyle='--',
        marker='s',
        markersize=9,
        linewidth=2.4,
        markeredgewidth=1.0,
        label='Computing Time'
    )
    ax2.set_ylabel("Computing Time (s)", fontweight='bold')
    ax2.tick_params(axis='both', width=1.2)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(lines, labels, loc="best", frameon=True)
    for txt in legend.get_texts():
        txt.set_fontweight('bold')

    plt.title(
        "Sampling Points vs Error and Computing Time",
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_collocation_vs_error(
    results,
    save_path="collocation_vs_error.pdf"
):
    sampling_points = [r["sampling_points"] for r in results]
    rel_err_vals = [r["rel_err"] * 100.0 for r in results]

    plt.figure(figsize=(7.8, 5.6))
    plt.plot(
        sampling_points, rel_err_vals,
        marker='o', markersize=9, linewidth=2.2
    )
    plt.xlabel("Number of Sampling Points", fontweight='bold')
    plt.ylabel("Relative L2 Error (%)", fontweight='bold')
    plt.title("Sampling Points vs Error", fontweight='bold')
    plt.grid(alpha=0.28, linestyle='--')
    # plt.xticks(fontweight='bold')
    # plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


# =========================================================
# Print Tables
# =========================================================
def print_hidden_feature_table(results):
    print("\n=== Hidden Feature Study Results ===")
    print(f"{'Features':>10} | {'RelL2(%)':>12} | {'Compute(s)':>12}")
    print("-" * 42)
    for r in results:
        print(
            f"{r['num_neuron']:>10d} | "
            f"{100 * r['rel_err']:>12.4f} | "
            f"{r['computing_time']:>12.4f}"
        )


def print_collocation_table(results):
    print("\n=== Collocation Study Results ===")
    print(
        f"{'N_pde':>8} | {'N_bnd':>8} | {'N_ini':>8} | "
        f"{'SamplePts':>10} | {'RelL2(%)':>10} | {'Compute(s)':>12}"
    )
    print("-" * 78)
    for r in results:
        print(
            f"{r['N_pde']:>8d} | "
            f"{r['N_bnd']:>8d} | "
            f"{r['N_ini']:>8d} | "
            f"{r['sampling_points']:>10d} | "
            f"{100 * r['rel_err']:>10.4f} | "
            f"{r['computing_time']:>12.4f}"
        )


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # Study 1: hidden features
    # -----------------------------------------------------
    neuron_list = [1000, 2000, 3000, 5000, 7000, 10000]
    hidden_results = study_hidden_features(
        neuron_list=neuron_list,
        fixed_N_pde=30000,
        fixed_N_bnd=20000,
        fixed_N_ini=10000,
        seed=42
    )
    print_hidden_feature_table(hidden_results)
    plot_hidden_features_vs_error_time(
        hidden_results,
        save_path="hidden_features_vs_error_time.pdf"
    )

    # -----------------------------------------------------
    # Study 2: collocation points
    # -----------------------------------------------------
    collocation_scales = [0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5]
    collocation_results = study_collocation_points(
        collocation_scales=collocation_scales,
        base_N_pde=30000,
        base_N_bnd=20000,
        base_N_ini=10000,
        fixed_num_neuron=5000,
        seed=42
    )
    print_collocation_table(collocation_results)
    plot_collocation_vs_error(
        collocation_results,
        save_path="collocation_vs_error.pdf"
    )

    plot_collocation_vs_error_time(
        collocation_results,
        save_path="collocation_vs_error_time.pdf"
    )
