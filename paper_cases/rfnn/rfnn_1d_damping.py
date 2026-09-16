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
from numba import njit

# ========== 1. Configuration & Setup ==========
initial_seed = 5
torch.manual_seed(initial_seed)
np.random.seed(initial_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# =========================
USE_PML = True

c = 1.0                 # Wave speed
L = 5.0                 # Physical domain [0, L]
T = 5.0                 # Total simulation time
num_neuron = 3000       # Number of neurons (Random Features)
r_neuron = [5, 5]       # Range for weight initialization

# ---------- Absorbing Boundary Settings (PML) ----------
pml_width = 1.5
sigma_max = 20.0
sigma_order = 3

# ---------- Source Settings ----------
x_src = L / 2.0
freq = 1.0

# ---------- Collocation Point Weights ----------
p_pde = 1.0
p_bnd = 10.0
p_ini_u = 10.0
p_ini_v = 10.0

# ---------- Training Point Counts ----------
N_pde = 30000
N_bnd = 20000
N_ini = 10000

# ---------- Reflection Coefficient Probe Positions ----------
probe_positions = [0.1, 0.2, 4.8, 4.9]

# ---------- Plotting Sampling Grid ----------
dx_plot = 0.01
dt_plot = dx_plot / c

# ---------- Reference Solution FDTD Parameters ----------
ref_dx = 1e-2
ref_CFL = 1.0
ref_sigma = 0.1
ref_omega = 2.0 * np.pi * freq

# ---------- Block-Solving Parameters ----------
solve_block_size = 5000       # Block size when assembling normal equations
predict_chunk_size_t = 20     # Chunk size for time-dimension during prediction

# ========== 2. Determine Computational Domain Based on Mode ==========
def get_domain_bounds(use_pml: bool):
    """
    Returns the computational domain bounds based on whether PML is used.
    With PML: domain extends beyond physical region.
    Without PML: domain is the physical region itself.
    """
    if use_pml:
        return -pml_width, L + pml_width
    else:
        return 0.0, L

x_left, x_right = get_domain_bounds(USE_PML)

# ========== 3. Print Grid Information ==========
def print_grid_info(name, dx_val, dt_val, Nx, Nt, x_min, x_max, t_min, t_max):
    """Prints detailed information about a given grid."""
    print("=" * 70)
    print(f"{name} Grid Information:")
    print(f"dx = {dx_val:.6f}, dt = {dt_val:.6f}")
    print(f"Spatial range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"Time range: [{t_min:.2f}, {t_max:.2f}]")
    print(f"Spatial points (Nx): {Nx}")
    print(f"Time points (Nt): {Nt}")
    print(f"CFL number: {c * dt_val / dx_val:.6f}")
    print("=" * 70)

# ========== 4. Energy Calculation ==========
def compute_energy(u_field, dx):
    """
    Calculates the total energy in the domain at each time step.
    u_field: shape = (Nx, Nt)
    dx: Real spatial integration step size.
    """
    return np.sum(u_field ** 2, axis=0) * dx

# ========== 5. Activation Function and Derivatives ==========
def atv(x):
    """Hyperbolic tangent activation function."""
    return torch.tanh(x)

def diff1_atv(x):
    """First derivative of tanh."""
    return 1.0 - torch.tanh(x) ** 2

def diff2_atv(x):
    """Second derivative of tanh."""
    th = torch.tanh(x)
    return 2.0 * th * (th**2 - 1.0)

# ========== 6. Damping/Sigma Function for PML ==========
def sigma_x(X):
    """
    Calculates the damping coefficient sigma(x) for the PML.
    Only active if USE_PML is True.
    """
    X = X.to(dtype)
    sig = torch.zeros_like(X, dtype=dtype, device=X.device)

    if not USE_PML:
        return sig

    # Apply damping to left PML region
    left_mask = X < 0.0
    if left_mask.any():
        sig[left_mask] = sigma_max * ((0.0 - X[left_mask]) / pml_width) ** sigma_order

    # Apply damping to right PML region
    right_mask = X > L
    if right_mask.any():
        sig[right_mask] = sigma_max * ((X[right_mask] - L) / pml_width) ** sigma_order

    return sig

# ========== 7. Training Data Generation ==========
def generate_points():
    """
    Generates collocation points for PDE, boundary, and initial conditions.
    Uses a mix of uniform and Gaussian sampling for PDE points.
    """
    N_uniform = N_pde
    N_gaussian = N_pde - N_uniform

    # Uniform sampling for PDE points across the entire domain
    xi_uniform = x_left + (x_right - x_left) * torch.rand(
        N_uniform, 1, device=device, dtype=dtype
    )
    ti_uniform = T * torch.rand(
        N_uniform, 1, device=device, dtype=dtype
    )

    # Gaussian sampling around the source location for higher resolution
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

    # Boundary points (on the extended domain if PML is used)
    tb = T * torch.rand(N_bnd, 1, device=device, dtype=dtype)
    ptb_left = torch.cat((torch.full_like(tb, x_left), tb), dim=1)
    ptb_right = torch.cat((torch.full_like(tb, x_right), tb), dim=1)
    ptb = torch.cat((ptb_left, ptb_right), dim=0).to(dtype)

    # Initial condition points (t=0)
    xi_ini = x_left + (x_right - x_left) * torch.rand(
        N_ini, 1, device=device, dtype=dtype
    )
    ti_ini = torch.zeros(N_ini, 1, device=device, dtype=dtype)
    pt_ini = torch.cat((xi_ini, ti_ini), dim=1).to(dtype)

    return pti, ptb, pt_ini

# ========== 8. Feature Value and Derivative Matrices ==========
def feature_value(W, b, points):
    """Computes the value of the basis function: tanh(W*X + b)."""
    points = points.to(W.dtype)
    b = b.to(W.dtype)

    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]

    Z = X @ W_x + Tt @ W_t + b
    return atv(Z)

def feature_t(W, b, points):
    """Computes the time derivative of the basis function."""
    points = points.to(W.dtype)
    b = b.to(W.dtype)

    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]

    Z = X @ W_x + Tt @ W_t + b
    return W_t * diff1_atv(Z)

def feature_xx(W, b, points):
    """Computes the second spatial derivative of the basis function."""
    points = points.to(W.dtype)
    b = b.to(W.dtype)

    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]

    Z = X @ W_x + Tt @ W_t + b
    return (W_x ** 2) * diff2_atv(Z)

def feature_tt(W, b, points):
    """Computes the second time derivative of the basis function."""
    points = points.to(W.dtype)
    b = b.to(W.dtype)

    X = points[:, 0:1]
    Tt = points[:, 1:2]
    W_x = W[0:1, :]
    W_t = W[1:2, :]

    Z = X @ W_x + Tt @ W_t + b
    return (W_t ** 2) * diff2_atv(Z)

# ========== 9. Source Term ==========
def source_term(points):
    """
    Defines the source term: cos(2*pi*f*t) * exp(-(x-x_src)^2 / 0.02).
    """
    points = points.to(dtype)
    X = points[:, 0:1]
    Tt = points[:, 1:2]
    return torch.cos(2.0 * np.pi * freq * Tt) * torch.exp(-((X - x_src) ** 2) / 0.02)

# ========== 10. Initial Conditions ==========
def u0(x):
    """Initial displacement: u(x, 0) = 0."""
    x = x.to(dtype)
    return torch.zeros_like(x)

def v0(x):
    """Initial velocity: du/dt(x, 0) = 0."""
    x = x.to(dtype)
    return torch.zeros_like(x)

# ========== 11. PDE Residual Matrix (Core Operator) ==========
def wave_equation_residual(W, b, points, c):
    """
    Computes the residual of the wave equation.
    If PML is enabled, adds the damping term sigma*u_t.
    """
    points = points.to(W.dtype)
    phi_tt = feature_tt(W, b, points)
    phi_xx = feature_xx(W, b, points)

    if USE_PML:
        X = points[:, 0:1]
        sig = sigma_x(X).to(W.dtype)
        phi_t = feature_t(W, b, points)
        return phi_tt + sig * phi_t - (c ** 2) * phi_xx
    else:
        return phi_tt - (c ** 2) * phi_xx

# ========== 12. Utility Functions for Comparison ==========
def interpolate_trace_along_x(x_grid, u_xt, x_query):
    """
    Interpolates the field u(x,t) at a specific x_query over all times.
    """
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
    """
    Interpolates a time trace from an old time grid to a new one.
    """
    return np.interp(t_new, t_old, trace_old)

def compute_reflection_metrics(
    x_num, t_num, u_num_xt,
    x_ref, t_ref, u_ref_xt,
    probe_positions,
    eps=1e-15
):
    """
    Compares numerical solution against a reference solution at specified probe points.
    Calculates reflection coefficients.
    """
    results = []

    for xpos in probe_positions:
        trace_num = interpolate_trace_along_x(x_num, u_num_xt, xpos)
        trace_ref_raw = interpolate_trace_along_x(x_ref, u_ref_xt, xpos)

        # Interpolate reference trace to match numerical solution's time grid if needed
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
            'x': xpos,
            'R_linear': R,
            'R_dB': R_dB,
            'trace_num': trace_num,
            'trace_ref': trace_ref,
            'trace_diff': diff_trace
        })

    return results

def compute_worst_reflection_in_region(
    x_num, t_num, u_num_xt,
    x_ref, t_ref, u_ref_xt,
    x_region_min, x_region_max,
    n_probe=50,
    eps=1e-15
):
    """
    Finds the point with the highest reflection coefficient within a specified region.
    """
    probe_x = np.linspace(x_region_min, x_region_max, n_probe)
    results = compute_reflection_metrics(
        x_num, t_num, u_num_xt,
        x_ref, t_ref, u_ref_xt,
        probe_x, eps=eps
    )
    valid = [r for r in results if np.isfinite(r['R_linear'])]
    if len(valid) == 0:
        return None
    return max(valid, key=lambda d: d['R_linear'])

# ========== 13. Reference Solution Generation (FDTD) ==========
def build_reference_problem(
    c, L_phys, T_total, dx, CFL, x_src_phys, sigma, omega,
    n_save_t=None, n_save_x=None
):
    """
    Sets up parameters for the FDTD reference solver.
    Creates an extended domain with PML-like absorbing layers.
    """
    dt = CFL * dx / c
    nt = int(np.ceil(T_total / dt))
    dt = T_total / nt
    cfl_actual = c * dt / dx
    if cfl_actual > 1.0 + 1e-12:
        raise ValueError(f"CFL violated in reference solver: {cfl_actual:.6f}")

    if n_save_t is None:
        n_save_t = nt + 1
    if n_save_x is None:
        n_save_x = int(round(L_phys / dx)) + 1

    # Calculate necessary guard width to prevent reflections from boundaries
    min_guard = max(0.0, c * T_total - L_phys / 2.0)
    safety_margin = max(6.0 * sigma, 20.0 * dx, 0.5)
    guard_width = min_guard + safety_margin

    L_ext = L_phys + 2.0 * guard_width
    nx_ext = int(round(L_ext / dx)) + 1
    x_ext = np.linspace(0.0, L_ext, nx_ext)

    x_phys_left = guard_width
    x_src_ext = x_phys_left + x_src_phys

    # Gaussian source profile on the extended grid
    gaussian_source_weights = np.exp(-(x_ext - x_src_ext) ** 2 / (2.0 * sigma ** 2))

    x_save_phys = np.linspace(0.0, L_phys, n_save_x)
    save_t_idx = np.arange(n_save_t, dtype=np.int64)

    return {
        'dt': dt,
        'nt': nt,
        'nx_ext': nx_ext,
        'x_phys_left': x_phys_left,
        'gaussian_source_weights': gaussian_source_weights,
        'x_save_phys': x_save_phys,
        'save_t_idx': save_t_idx
    }

@njit
def fdtd_1d_reference_main_only(
    c, dx, dt, nt, nx_ext,
    omega, gaussian_source_weights,
    x_phys_left, x_save_phys, save_t_idx
):
    """
    Fast FDTD solver implemented with Numba for efficiency.
    Solves on an extended grid and saves only the physical domain.
    """
    coef = (c * dt / dx) ** 2

    E_prev = np.zeros(nx_ext)
    E_current = np.zeros(nx_ext)
    E_next = np.zeros(nx_ext)

    n_save_t = save_t_idx.shape[0]
    n_save_x = x_save_phys.shape[0]
    E_save_phys = np.zeros((n_save_t, n_save_x))

    save_counter = 0

    for n in range(nt + 1):
        t_now = n * dt

        # Update interior points
        for i in range(1, nx_ext - 1):
            E_next[i] = (
                2.0 * E_current[i]
                - E_prev[i]
                + coef * (E_current[i + 1] - 2.0 * E_current[i] + E_current[i - 1])
            )

        # Apply source term
        amplitude = 0.5 if n < 1 else 1.0
        source_factor = amplitude * (dt ** 2) * np.cos(omega * t_now)
        for i in range(nx_ext):
            E_next[i] += gaussian_source_weights[i] * source_factor

        # Apply Dirichlet boundary conditions (hard walls)
        E_next[0] = 0.0
        E_next[nx_ext - 1] = 0.0

        # Save solution for physical domain only
        if save_counter < n_save_t and n == save_t_idx[save_counter]:
            for j in range(n_save_x):
                x_abs = x_phys_left + x_save_phys[j]
                cell = x_abs / dx
                i0 = int(np.floor(cell))

                if i0 < 0:
                    i0 = 0
                if i0 >= nx_ext - 1:
                    i0 = nx_ext - 2

                alpha = cell - i0
                E_save_phys[save_counter, j] = (
                    (1.0 - alpha) * E_current[i0] + alpha * E_current[i0 + 1]
                )
            save_counter += 1

        # Swap arrays for next time step
        tmp = E_prev
        E_prev = E_current
        E_current = E_next
        E_next = tmp

        for i in range(nx_ext):
            E_next[i] = 0.0

    return E_save_phys

def generate_reference_solution():
    """
    Generates the reference solution using the FDTD method.
    """
    ref_cfg = build_reference_problem(
        c=c,
        L_phys=L,
        T_total=T,
        dx=ref_dx,
        CFL=ref_CFL,
        x_src_phys=x_src,
        sigma=ref_sigma,
        omega=ref_omega,
        n_save_t=None,
        n_save_x=None
    )

    # Warm-up run for Numba
    _ = fdtd_1d_reference_main_only(
        c, ref_dx, ref_cfg['dt'], 1, ref_cfg['nx_ext'],
        ref_omega, ref_cfg['gaussian_source_weights'],
        ref_cfg['x_phys_left'], ref_cfg['x_save_phys'],
        np.array([0, 1], dtype=np.int64)
    )

    start = time.perf_counter()
    E_save_phys = fdtd_1d_reference_main_only(
        c, ref_dx, ref_cfg['dt'], ref_cfg['nt'], ref_cfg['nx_ext'],
        ref_omega, ref_cfg['gaussian_source_weights'],
        ref_cfg['x_phys_left'], ref_cfg['x_save_phys'],
        ref_cfg['save_t_idx']
    )
    elapsed = time.perf_counter() - start

    x_ref = ref_cfg['x_save_phys']
    t_ref = ref_cfg['save_t_idx'] * ref_cfg['dt']
    u_ref_xt = E_save_phys.T  # Transpose to (Nx, Nt)

    print(f"Reference solution generated in {elapsed:.3f} s")
    return x_ref, t_ref, u_ref_xt

# ========== 14. Block-Wise Normal Equation Solver ==========
@torch.no_grad()
def solve_blockwise_normal_equation(W, b, pti, ptb, pt_ini, c, block_size=4000, ridge=1e-6):
    """
    Solves the normal equation A^T A Ur = A^T F in blocks to avoid memory overflow.
    The PDE part is accumulated in blocks, while boundary/initial parts are accumulated once.
    """
    M = W.shape[1]
    ATA = torch.zeros((M, M), dtype=dtype, device=device)
    ATF = torch.zeros((M, 1), dtype=dtype, device=device)

    n_pde_total = pti.shape[0]

    start_solve = time.perf_counter()

    # -------------------------------------------------
    # 1) Accumulate PDE contributions in blocks
    # -------------------------------------------------
    for i0 in range(0, n_pde_total, block_size):
        i1 = min(i0 + block_size, n_pde_total)
        pti_block = pti[i0:i1]

        A_pde_block = p_pde * wave_equation_residual(W, b, pti_block, c)
        F_pde_block = p_pde * source_term(pti_block).to(dtype)

        ATA += A_pde_block.T @ A_pde_block
        ATF += A_pde_block.T @ F_pde_block

        # Free memory after processing each block
        del pti_block, A_pde_block, F_pde_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------
    # 2) Accumulate boundary condition contribution once
    # -------------------------------------------------
    A_bnd = p_bnd * feature_value(W, b, ptb)
    F_bnd = torch.zeros((ptb.shape[0], 1), dtype=dtype, device=device)

    ATA += A_bnd.T @ A_bnd
    ATF += A_bnd.T @ F_bnd

    del A_bnd, F_bnd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 3) Accumulate initial displacement contribution once
    # -------------------------------------------------
    A_ini_u = p_ini_u * feature_value(W, b, pt_ini)
    F_ini_u = p_ini_u * u0(pt_ini[:, 0:1]).to(dtype)

    ATA += A_ini_u.T @ A_ini_u
    ATF += A_ini_u.T @ F_ini_u

    del A_ini_u, F_ini_u
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 4) Accumulate initial velocity contribution once
    # -------------------------------------------------
    A_ini_v = p_ini_v * feature_t(W, b, pt_ini)
    F_ini_v = p_ini_v * v0(pt_ini[:, 0:1]).to(dtype)

    ATA += A_ini_v.T @ A_ini_v
    ATF += A_ini_v.T @ F_ini_v

    del A_ini_v, F_ini_v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 5) Regularize and solve the final system
    # -------------------------------------------------
    ATA += ridge * torch.eye(M, dtype=dtype, device=device)
    Ur = torch.linalg.solve(ATA, ATF)

    elapsed = time.perf_counter() - start_solve
    print(f"Blockwise normal-equation solve finished in {elapsed:.3f} s")

    return Ur

# ========== 15. Batched Prediction ==========
@torch.no_grad()
def predict_main_grid_in_batches(x_test, t_test, W, b, Ur, chunk_size_t=20):
    """
    Performs prediction on the main domain grid in time chunks to manage memory.
    Returns a numpy array of shape (Nx, Nt).
    """
    Nx = x_test.shape[0]
    Nt = t_test.shape[0]

    u_pred_np = np.zeros((Nx, Nt), dtype=np.float64)

    W_x = W[0:1, :]   # (1, M)
    W_t = W[1:2, :]   # (1, M)

    # Pre-compute the spatial part of the feature map
    X_part = x_test.unsqueeze(1) @ W_x  # (Nx, M)

    pred_start = time.perf_counter()

    for j0 in range(0, Nt, chunk_size_t):
        j1 = min(j0 + chunk_size_t, Nt)
        t_chunk = t_test[j0:j1]

        # Compute the time part of the feature map for this chunk
        T_part = t_chunk.unsqueeze(1) @ W_t                 # (chunk, M)
        # Broadcast spatial and temporal parts together
        Z_chunk = X_part[:, None, :] + T_part[None, :, :] + b[None, :, :]
        phi_chunk = atv(Z_chunk)

        # Reshape for batched matrix multiplication with Ur
        phi_2d = phi_chunk.reshape(-1, phi_chunk.shape[-1])  # (Nx*chunk, M)
        u_chunk = (phi_2d @ Ur).reshape(Nx, j1 - j0)

        u_pred_np[:, j0:j1] = u_chunk.detach().cpu().numpy()

        # Clean up temporary variables
        del t_chunk, T_part, Z_chunk, phi_chunk, phi_2d, u_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - pred_start
    print(f"Batched prediction finished in {elapsed:.3f} s")

    return u_pred_np

# ========== 16. Build Plotting Indices ==========
def build_plot_indices_from_physical_step(x_grid, t_grid, dx_target, dt_target):
    """
    Selects uniform indices from existing grids based on desired physical step sizes.
    This allows for coarser plotting without regenerating the full solution.
    """
    x_min, x_max = x_grid[0], x_grid[-1]
    t_min, t_max = t_grid[0], t_grid[-1]

    # Calculate desired number of points
    Nx_plot = int(round((x_max - x_min) / dx_target)) + 1
    Nt_plot = int(round((t_max - t_min) / dt_target)) + 1

    # Find the closest indices in the original grid
    x_plot_target = np.linspace(x_min, x_max, Nx_plot)
    t_plot_target = np.linspace(t_min, t_max, Nt_plot)

    x_idx = np.searchsorted(x_grid, x_plot_target)
    x_idx = np.clip(x_idx, 0, len(x_grid) - 1)

    t_idx = np.searchsorted(t_grid, t_plot_target)
    t_idx = np.clip(t_idx, 0, len(t_grid) - 1)

    # Remove duplicates caused by rounding
    x_idx = np.unique(x_idx)
    t_idx = np.unique(t_idx)

    return x_idx, t_idx

# ========== 17. Main Execution ==========
# ---- Initialize Random Feature Parameters ----
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

# Initialize biases and apply bias correction
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

# ---- Generate Training Points ----
pti, ptb, pt_ini = generate_points()

pti = pti.to(dtype)
ptb = ptb.to(dtype)
pt_ini = pt_ini.to(dtype)
W = W.to(dtype)
b = b.to(dtype)

print(f"USE_PML: {USE_PML}")
print(f"Domain: [{x_left}, {x_right}]")
print(f"Device: {device}")

# ========== Generate Reference Solution First ==========
x_ref, t_ref, u_ref_xt = generate_reference_solution()
# Optionally load a pre-saved reference instead
# u_ref_xt = np.load('1d_fdtd_ref_5e-5.npy').T

dx_pred = x_ref[1] - x_ref[0]
dt_pred = t_ref[1] - t_ref[0]
Nx_pred = len(x_ref)
Nt_pred = len(t_ref)

print_grid_info(
    name="Prediction / Reference",
    dx_val=dx_pred,
    dt_val=dt_pred,
    Nx=Nx_pred,
    Nt=Nt_pred,
    x_min=x_ref[0],
    x_max=x_ref[-1],
    t_min=t_ref[0],
    t_max=t_ref[-1]
)

total_start_time = time.time()

# ========== Solve in Blocks ==========
Ur = solve_blockwise_normal_equation(
    W=W,
    b=b,
    pti=pti,
    ptb=ptb,
    pt_ini=pt_ini,
    c=c,
    block_size=solve_block_size,
    ridge=1e-6
)

# ========== Predict RFNN Solution on Reference Grid ==========
x_test_main = torch.tensor(x_ref, device=device, dtype=dtype)
t_test_main = torch.tensor(t_ref, device=device, dtype=dtype)

u_pred_main = predict_main_grid_in_batches(
    x_test=x_test_main,
    t_test=t_test_main,
    W=W,
    b=b,
    Ur=Ur,
    chunk_size_t=predict_chunk_size_t
)

print("Prediction Finished")
total_end_time = time.time()
print(f"\nTotal wall-clock time: {total_end_time - total_start_time:.3f} s")

# ========== Compare with Reference Solution ==========
rel_err = np.linalg.norm(u_pred_main - u_ref_xt) / max(np.linalg.norm(u_ref_xt), 1e-15)
print(f"Relative L2 Error (main domain, vs reflection-free reference): {rel_err:.2%}")

error_main = u_pred_main - u_ref_xt

# Calculate reflection metrics
reflection_results = compute_reflection_metrics(
    x_num=x_ref,
    t_num=t_ref,
    u_num_xt=u_pred_main,
    x_ref=x_ref,
    t_ref=t_ref,
    u_ref_xt=u_ref_xt,
    probe_positions=probe_positions
)

print("\n===== Reflection Coefficients at Probe Points =====")
for item in reflection_results:
    print(
        f"x = {item['x']:.2f} : "
        f"R = {item['R_linear']:.6e}, "
        f"R_dB = {item['R_dB']:.2f} dB"
    )

print("\n===== Summary Table =====")
print(f"{'Probe x':>10s} | {'R (linear)':>14s} | {'R (dB)':>10s}")
print("-" * 42)
for item in reflection_results:
    print(f"{item['x']:10.2f} | {item['R_linear']:14.6e} | {item['R_dB']:10.2f}")

# Find worst reflection on left and right sides
left_worst = compute_worst_reflection_in_region(
    x_ref, t_ref, u_pred_main,
    x_ref, t_ref, u_ref_xt,
    0.0, 0.5, n_probe=50
)
right_worst = compute_worst_reflection_in_region(
    x_ref, t_ref, u_pred_main,
    x_ref, t_ref, u_ref_xt,
    4.5, 5.0, n_probe=50
)

if left_worst is not None:
    print(f"\nWorst reflection on left side [0.0, 0.5]: x = {left_worst['x']:.4f}, "
          f"R = {left_worst['R_linear']:.6e}, R_dB = {left_worst['R_dB']:.2f} dB")
if right_worst is not None:
    print(f"Worst reflection on right side [4.5, 5.0]: x = {right_worst['x']:.4f}, "
          f"R = {right_worst['R_linear']:.6e}, R_dB = {right_worst['R_dB']:.2f} dB")

# ========== Prepare Data for Plotting ==========
x_idx_plot, t_idx_plot = build_plot_indices_from_physical_step(
    x_grid=x_ref,
    t_grid=t_ref,
    dx_target=dx_plot,
    dt_target=dt_plot
)

x_plot = x_ref[x_idx_plot]
t_plot = t_ref[t_idx_plot]

u_plot = u_pred_main[np.ix_(x_idx_plot, t_idx_plot)]
u_plot_xt = u_plot.T

u_ref_plot = u_ref_xt[np.ix_(x_idx_plot, t_idx_plot)]
error_plot = error_main[np.ix_(x_idx_plot, t_idx_plot)]

actual_dx_plot = x_plot[1] - x_plot[0] if len(x_plot) > 1 else np.nan
actual_dt_plot = t_plot[1] - t_plot[0] if len(t_plot) > 1 else np.nan

print_grid_info(
    name="Plot",
    dx_val=actual_dx_plot,
    dt_val=actual_dt_plot,
    Nx=len(x_plot),
    Nt=len(t_plot),
    x_min=x_plot[0],
    x_max=x_plot[-1],
    t_min=t_plot[0],
    t_max=t_plot[-1]
)

# ========== Calculate Energy (using real prediction grid dx_pred) ==========
energy = compute_energy(u_pred_main, dx_pred)

# ========== Visualization ==========
plt.figure(figsize=(12, 8))

# ---- Subplot 1: RFNN Spacetime Field ----
plt.subplot(2, 2, 1)
plt.pcolormesh(x_plot, t_plot, u_plot_xt, shading='auto', cmap='RdBu')
plt.colorbar(label='u(x,t)')
plt.xlabel('Position x')
plt.ylabel('Time t')
plt.title(
    f'RFNN wave field in main domain\n'
    f'prediction dx={dx_pred:.4e}, plot dx={actual_dx_plot:.4e}'
)
plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.7, label='Boundary')
plt.axvline(x=L, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=x_src, color='g', linestyle='-', alpha=0.7, label='Source')
plt.legend()

# ---- Subplot 2: Cross-sections at specific times ----
plt.subplot(2, 2, 2)
time_indices = [
    int(0.1 * len(t_plot)),
    int(0.3 * len(t_plot)),
    int(0.5 * len(t_plot)),
    int(0.7 * len(t_plot))
]
colors = ['b', 'r', 'g', 'm']
for i, tidx in enumerate(time_indices):
    plt.plot(x_plot, u_plot_xt[tidx, :], color=colors[i], label=f't = {t_plot[tidx]:.2f}')
plt.xlabel('Position x')
plt.ylabel('u(x,t)')
plt.title('Snapshots in main domain')
plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=L, color='r', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend()

# ---- Subplot 3: Time evolution at specific positions ----
plt.subplot(2, 2, 3)
position_list = [0.5, 2.5, 4.5]
for i, xpos in enumerate(position_list):
    x_idx = np.argmin(np.abs(x_ref - xpos))
    plt.plot(t_ref, u_pred_main[x_idx, :], color=colors[i], label=f'x = {x_ref[x_idx]:.2f}')
plt.xlabel('Time t')
plt.ylabel('u(x,t)')
plt.title('Time evolution at different positions')
plt.grid(True, alpha=0.3)
plt.legend()

# ---- Subplot 4: Total Energy Evolution ----
plt.subplot(2, 2, 4)
plt.semilogy(t_ref, energy, 'b-', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Total Energy (log scale)')
plt.title(f'Energy evolution in main domain\n(using true dx={dx_pred:.6e})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== Main Domain Error Spacetime Plot ==========
plt.figure(figsize=(10, 6))
plt.pcolormesh(x_plot, t_plot, error_plot.T, shading='auto', cmap='RdBu')
plt.colorbar(label='u_RFNN - u_ref')
plt.xlabel('Position x in main domain')
plt.ylabel('Time t')
plt.title(
    f'Error field in main domain (RFNN - reflection-free reference)\n'
    f'prediction dx={dx_pred:.4e}, plot dx={actual_dx_plot:.4e}'
)
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=L, color='k', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ========== Time Series Comparison at Probe Points ==========
plt.figure(figsize=(12, 8))
for k, item in enumerate(reflection_results):
    plt.subplot(2, 2, k + 1)
    plt.plot(t_ref, item['trace_num'], label='RFNN', linewidth=2)
    plt.plot(t_ref, item['trace_ref'], '--', label='Reference', linewidth=2)
    plt.plot(t_ref, item['trace_diff'], ':', label='Difference', linewidth=1.5)
    plt.xlabel('Time t')
    plt.ylabel('Field')
    plt.title(f"x = {item['x']:.2f}, R_dB = {item['R_dB']:.2f} dB")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()
