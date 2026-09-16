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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib.ticker import ScalarFormatter

# ========== Parameter Configuration ==========
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c = 1.0              # Wave speed
L = 5.0              # Spatial range [0, L] (for x and y dimensions)
T = 1.0              # Time range [0, T]
num_neuron = 10000   # Number of neurons
r_neuron = [5, 4, 4] # Weight initialization range (added y dimension)
d = 3                # Input dimension (t, x, y)
sigma = 0.2

# ========== Helper Functions ==========
atv = lambda x: torch.tanh(x)
diff1_atv = lambda x: 1 - torch.tanh(x)**2
diff2_atv = lambda x: 2 * torch.tanh(x) * (torch.tanh(x)**2 - 1)

# ========== Training Data Generation ==========
def generate_points():
    N_pde = 20000
    N_bnd = 10000
    N_ini = 10000

    # Internal point sampling (3D space)
    ti = torch.rand(N_pde, 1, device=device) * T
    xi = torch.rand(N_pde, 1, device=device) * L
    yi = torch.rand(N_pde, 1, device=device) * L
    pti = torch.cat((ti, xi, yi), dim=1)

    # Boundary conditions (x=0, x=L, y=0, y=L)
    tb = torch.rand(N_bnd, 1, device=device) * T
    xb = torch.rand(N_bnd, 1, device=device) * L
    yb = torch.rand(N_bnd, 1, device=device) * L
    # x boundaries
    ptb_x0 = torch.cat((tb, torch.zeros_like(tb), yb), dim=1)
    ptb_xL = torch.cat((tb, torch.full_like(tb, L), yb), dim=1)
    # y boundaries
    ptb_y0 = torch.cat((tb, xb, torch.zeros_like(tb)), dim=1)
    ptb_yL = torch.cat((tb, xb, torch.full_like(tb, L)), dim=1)

    ptb = torch.cat((ptb_x0, ptb_xL, ptb_y0, ptb_yL), dim=0)

    # Initial conditions
    ti_ini = torch.zeros(N_ini, 1, device=device)
    xi_ini = torch.rand(N_ini, 1, device=device) * L
    yi_ini = torch.rand(N_ini, 1, device=device) * L
    pt_ini = torch.cat((ti_ini, xi_ini, yi_ini), dim=1)

    return pti, ptb, pt_ini

# ========== Physics Constraint Definitions ==========
def wave_equation_residual(W, b, points, c):
    T = points[:, 0:1]   # Time coordinate
    X = points[:, 1:2]   # x coordinate
    Y = points[:, 2:3]   # y coordinate

    W_t = W[0:1, :]      # Time weight
    W_x = W[1:2, :]      # x spatial weight
    W_y = W[2:3, :]      # y spatial weight

    Z = T@W_t + X@W_x + Y@W_y + b

    # Second-order derivative terms
    phi_tt = torch.sum(W_t**2, 0) * diff2_atv(Z)
    phi_xx = torch.sum(W_x**2, 0) * diff2_atv(Z)
    phi_yy = torch.sum(W_y**2, 0) * diff2_atv(Z)

    return phi_tt - (c**2)*(phi_xx + phi_yy)

def ini_residual(W, b, points):
    T = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]

    W_t = W[0:1, :]
    W_x = W[1:2, :]
    W_y = W[2:3, :]

    Z = T@W_t + X@W_x + Y@W_y + b
    phi_t = W_t * (1 - atv(Z)**2)

    return phi_t

# ========== Main Solver Function ==========
start_time = time.time()

# Initialize weights and biases (added y dimension)
W1 = torch.tensor(np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)), dtype=torch.float32)
W2 = torch.tensor(np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)), dtype=torch.float32)
W3 = torch.tensor(np.random.uniform(-r_neuron[2], r_neuron[2], (1, num_neuron)), dtype=torch.float32)
W = torch.cat((W1, W2, W3), dim=0).to(device)

b1 = torch.tensor(np.random.uniform(0, T, (1, num_neuron)), dtype=torch.float32)
b2 = torch.tensor(np.random.uniform(0, L, (1, num_neuron)), dtype=torch.float32)
b3 = torch.tensor(np.random.uniform(0, L, (1, num_neuron)), dtype=torch.float32)
b = torch.cat((b1, b2, b3), dim=0).to(device)
b = -torch.sum(b * W, dim=0)

# Generate training points
pti, ptb, pt_ini = generate_points()

# Construct constraint matrix
A_pde = wave_equation_residual(W, b, pti, c)
A_bnd = atv(ptb @ W + b)  # Boundary condition
A_ini1 = atv(pt_ini @ W + b)  # Initial condition u
A_ini2 = ini_residual(W, b, pt_ini) # Initial condition du/dt

p_pde = 1
p_bnd = 10
p_ini = 10

A = torch.cat([p_pde*A_pde, p_bnd*A_bnd, p_ini*A_ini1, p_ini*A_ini2], dim=0)

# Right-hand side vector (source terms)
F_pde = torch.zeros_like(pti[:, 0:1])

# First source: cosine function, located at (x=1.5, y=1.5)
source_term_1 = torch.cos(2 * np.pi * pti[:, 0:1]) * torch.exp(-((pti[:, 1:2] - 1.5)**2 + (pti[:, 2:3] - 1.5)**2) / (2*sigma**2))

# Second source: temporal Gaussian pulse, peak at t=0.1, std dev 0.1, located at (x=4, y=4)
time_gaussian = torch.exp(-((pti[:, 0:1] - 0.1)**2) / (2 * 0.1**2))
space_gaussian = torch.exp(-((pti[:, 1:2] - 4.0)**2 + (pti[:, 2:3] - 4.0)**2) / (2*sigma**2))
source_term_2 = time_gaussian * space_gaussian

# Total source term is the sum of the two sources
F_pde += source_term_1 + source_term_2

F_bnd = torch.zeros_like(ptb[:, 0:1])
F_ini1 = torch.zeros_like(pt_ini[:, 0:1])
F_ini2 = torch.zeros_like(pt_ini[:, 0:1])

F = torch.cat([p_pde*F_pde, p_bnd*F_bnd, p_ini*F_ini1, p_ini*F_ini2], dim=0)

# Least squares solution
Ur = torch.linalg.lstsq(A, F)[0]
end_time = time.time()
print('Training time: ', end_time-start_time)

nt = 151
nx = 501
ny = 501

start_time = time.time()
t_test = torch.linspace(0, T, nt, device=device)
x_test = torch.linspace(0, L, nx, device=device)
y_test = torch.linspace(0, L, ny, device=device)

T_mesh, X, Y = torch.meshgrid(t_test, x_test, y_test)
test_points = torch.cat((T_mesh.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1).to(device)

# ========== Batch Prediction of Field Values ==========
batch_size = 20000  # Adjust this value based on your GPU memory
u_pred = []

with torch.no_grad():
    for i in range(0, test_points.shape[0], batch_size):
        batch = test_points[i:i+batch_size]
        # Incremental calculation for each batch
        hidden = torch.tanh(batch @ W + b)  # (batch_size, num_neuron)
        output = hidden @ Ur                # (batch_size, 1)
        u_pred.append(output.cpu())

u_pred = torch.cat(u_pred).numpy().reshape(X.shape)
u_analytical = np.load(DATA_DIR / '2d_fdtd_two_source_2e-3.npy')
error = u_pred - u_analytical
rel_err_rfnn_2 = np.linalg.norm(u_pred - u_analytical) / np.linalg.norm(u_analytical)
print(f"Relative L2 Error: {rel_err_rfnn_2:.2%}")

nt = 151
nx = 501
ny = 501

start_time = time.time()
t_test = torch.linspace(0, T, nt, device=device)
x_test = torch.linspace(0, L, nx, device=device)
y_test = torch.linspace(0, L, ny, device=device)

T_mesh, X, Y = torch.meshgrid(t_test, x_test, y_test)
test_points = torch.cat((T_mesh.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1).to(device)

# ========== Batch Prediction of Field Values ==========
batch_size = 20000  # Adjust this value based on your GPU memory
u_pred = []

with torch.no_grad():
    for i in range(0, test_points.shape[0], batch_size):
        batch = test_points[i:i+batch_size]
        # Incremental calculation for each batch
        hidden = torch.tanh(batch @ W + b)  # (batch_size, num_neuron)
        output = hidden @ Ur                # (batch_size, 1)
        u_pred.append(output.cpu())

end_time = time.time()
print('Prediction time: ', end_time-start_time)
u_pred = torch.cat(u_pred).numpy().reshape(X.shape)

# Dynamically calculate indices for given time points
target_times_ratio = [0.25, 0.5, 0.75, 1.0]  # Target times are 0.25, 0.5, 0.75, and 1.0 times T
target_times = np.array(target_times_ratio) * T
time_indices = np.array([37, 75, 112, 150])

# ========== Visualization ==========
def plot_snapshots_and_errors(time_indices):
    # Create a new figure and a 2x4 subplot grid
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize for better aspect ratio

    times = [f"t = {target_times[i]:.2f}s" for i in range(len(time_indices))]  # Use target times as titles

    # First row: Plot snapshots of u_pred_rfnn_2
    for idx, (ax, t_idx) in enumerate(zip(axs[0], time_indices)):
        im = ax.imshow(u_pred[t_idx].T,
                       extent=[0, L, 0, L],
                       origin='lower',
                       cmap='jet')
        ax.set_title(times[idx], fontweight='bold', fontsize=15)
        ax.set_xlabel('x', fontsize=15, fontweight='bold')
        ax.set_ylabel('y', fontsize=15, fontweight='bold')
        # Set font size for X and Y axis ticks
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        # Add color bar, use shrink parameter to adjust its size to match the image ratio
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=12)  # Set font size on the color bar

        # Set the color bar title and increase spacing
        cbar.ax.set_title('E(V/m)', fontweight='bold', fontsize=15, pad=20)  # Increase pad value

    # Second row: Plot snapshots of error_rfnn_2
    for idx, (ax, t_idx) in enumerate(zip(axs[1], time_indices)):
        im = ax.imshow(error[t_idx].T,
                       extent=[0, L, 0, L],
                       origin='lower',
                       cmap='coolwarm')  # Set maximum value for error
        ax.set_title(f"Error at {times[idx]}", fontweight='bold', fontsize=15)
        ax.set_xlabel('x', fontsize=15, fontweight='bold')
        ax.set_ylabel('y', fontsize=15, fontweight='bold')
        # Set font size for X and Y axis ticks
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        # Add color bar, use shrink parameter to adjust its size to match the image ratio
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=12)  # Set font size on the color bar

        # Set color bar to scientific notation, placing 10^n at the top
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Force scientific notation
        cbar.ax.yaxis.set_major_formatter(formatter)

        # Adjust the position of 10^n
        cbar.ax.yaxis.offsetText.set_fontsize(12)  # Set offset text font size
        cbar.ax.yaxis.offsetText.set_position((4.5, 1.1))  # Move 10^n up further

        # Set the color bar title and increase spacing
        cbar.ax.set_title('E(V/m)', fontweight='bold', fontsize=15, pad=20)  # Increase pad value

    # Add (a) and (b) below the first and second rows respectively
    fig.text(0.5, 0.48, '(a)', ha='center', va='center', fontsize=16, fontweight='bold')  # Below first row
    fig.text(0.5, 0.02, '(b)', ha='center', va='center', fontsize=16, fontweight='bold')  # Below second row

    # Set global title
    plt.suptitle('2D Wave Field Distribution and Errors from RFNN', fontsize=20, fontweight='bold')
    plt.tight_layout()  # Leave space for the global title
    plt.savefig('2D_errors_rfnn_0.2_two_sources.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# Call the function to plot snapshots and errors
plot_snapshots_and_errors(time_indices=time_indices)
