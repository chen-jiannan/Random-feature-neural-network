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
import time
from matplotlib.ticker import ScalarFormatter

# ========== Parameter Configuration ==========
initial_seed = 42  # Initial random seed for generating Ur
torch.manual_seed(initial_seed)
np.random.seed(initial_seed)  # Set initial random seed to ensure reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c = 1.0           # Wave speed
L = 5.0           # Spatial range [0, L]
T = 5.0           # Time range [0, T]
num_neuron = 5000 # Number of neurons (Random Features)
r_neuron = [5, 5] # Weight initialization range
d = 2             # Input dimension (t, x)

# ========== Helper Functions (Activation & Derivatives) ==========
# Tanh activation function
atv = lambda x: torch.tanh(x)
# First derivative of tanh: 1 - tanh^2(x)
diff1_atv = lambda x: 1 - torch.tanh(x) ** 2
# Second derivative of tanh: 2 * tanh(x) * (tanh^2(x) - 1)
diff2_atv = lambda x: 2 * torch.tanh(x) * (torch.tanh(x)**2 - 1)

# ========== Training Data Generation ==========
def generate_points():
    # Define number of collocation points for PDE, Boundary, and Initial conditions
    N_pde = 30000
    N_bnd = 20000
    N_ini = 10000

    # 1. PDE Residual Points (Interior of domain)
    # Uniformly sample t in [0, T] and x in [0, L]
    xi = torch.rand(N_pde, 1, device=device) * L
    ti = torch.rand(N_pde, 1, device=device) * T
    pti = torch.cat((ti, xi), dim=1) # Shape: (N_pde, 2)

    # 2. Boundary Condition Points (x=0 and x=L)
    tb = torch.rand(N_bnd, 1, device=device) * T
    # Left boundary (x=0)
    ptb_0 = torch.cat((tb, torch.zeros_like(tb)), dim=1)
    # Right boundary (x=L)
    ptb_L = torch.cat((tb, torch.full_like(tb, L)), dim=1)
    ptb = torch.cat((ptb_0, ptb_L), dim=0) # Combine boundaries

    # 3. Initial Condition Points (t=0)
    xi_ini = torch.rand(N_ini, 1, device=device) * L
    ti_ini = torch.zeros(N_ini, 1, device=device)
    pt_ini = torch.cat((ti_ini, xi_ini), dim=1)

    return pti, ptb, pt_ini

# ========== Physics Constraints Definition ==========
def wave_equation_residual(W, b, points, c):
    """
    Calculates the residual of the Wave Equation: u_tt - c^2 * u_xx = f
    Using automatic differentiation logic via the specific Random Feature structure.
    """
    T = points[:, 0:1]   # Time coordinate (N, 1)
    X = points[:, 1:2]   # Spatial coordinate (N, 1)

    # Separate weights corresponding to time and space inputs
    # W shape is (2, num_neuron). W[0] is for t, W[1] is for x
    W_t = W[0:1, :]     # Time weights (1, m)
    W_x = W[1:2, :]     # Space weights (1, m)

    # Calculate the feature map Z = W_t * t + W_x * x + b
    # Matrix multiplication: (N,1) @ (1,m) results in (N,m)
    Z = T @ W_t + X @ W_x + b

    # Calculate second derivatives using the chain rule
    # d^2/dt^2 tanh(Z) = W_t^2 * tanh''(Z)
    # We sum over the neuron dimension to aggregate contributions if necessary,
    # but here we keep the (N, m) shape to match the linear solver structure later.
    # Note: The original code sums over dim 0, resulting in (1, m), then broadcasts.
    # To ensure matrix multiplication compatibility later, we maintain shape consistency.

    # Second derivative components
    phi_tt = (W_t**2) * diff2_atv(Z) # Element-wise multiplication
    phi_xx = (W_x**2) * diff2_atv(Z)

    # Return the PDE residual operator applied to the basis functions: (L_phi_tt - c^2 * L_phi_xx)
    return phi_tt - (c**2)*phi_xx

def ini_residual(W, b, points):
    """
    Calculates the residual for the initial velocity condition: du/dt at t=0
    """
    T = points[:, 0:1]   # Time coordinate (N, 1)
    X = points[:, 1:2]   # Spatial coordinate (N, 1)

    # Separate weights
    W_t = W[0:1, :]
    W_x = W[1:2, :]

    # Feature map Z
    Z = T @ W_t + X @ W_x + b

    # First derivative with respect to time: du/dt = W_t * tanh'(Z)
    phi_t = W_t * diff1_atv(Z)

    return phi_t

def u(x):
    """Analytical function for initial displacement (sin(pi*x))"""
    return torch.sin(np.pi * x).reshape(-1, 1)

# ========== Main Solver Function ==========
start_time = time.time()
L_left = 0
L_right = L
T_left = 0
T_right = T

# Source Parameters
x_src = 2.5             # Source location (unused in the actual source term below, likely legacy)
sigma = 0.1             # Standard deviation of the Gaussian source
omega = 2 * np.pi       # Angular frequency (rad/s)

x_src2 = 4.0      # Second source center position
t_peak = 1.5      # Peak time

# Initialize Weights and Biases (Random Features)
# Weights W are fixed random numbers. Only the output coefficients Ur will be solved.
W1 = torch.tensor(np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)), dtype=torch.float32)
W2 = torch.tensor(np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)), dtype=torch.float32)
W = torch.cat((W1, W2), dim=0).to(device) # Shape: (2, num_neuron)

# Initialize biases.
# Note: The bias is subsequently modified to center the activation functions.
b1 = torch.tensor(np.random.uniform(L_left, L_right, (1, num_neuron)), dtype=torch.float32)
b2 = torch.tensor(np.random.uniform(T_left, T_right, (1, num_neuron)), dtype=torch.float32)
b = torch.cat((b1, b2), dim=0).to(device)

# Bias Correction:
# This step shifts the bias so that the random features are centered relative to the weights.
# b_new = - sum(b_old * W). This is a specific initialization strategy for RFM.
b = -torch.sum(b * W, dim=0)

# Generate Training Points
pti, ptb, pt_ini = generate_points()

# ========== Construct Constraint Matrices (The "A" Matrix) ==========
# We are building a linear system A * Ur = F, where Ur are the unknown output weights.

# 1. PDE Residual Matrix (Interior points)
# Shape: (N_pde, num_neuron)
A_pde = wave_equation_residual(W, b, pti, c)

# 2. Boundary Condition Matrix (Dirichlet: u=0)
# Shape: (2*N_bnd, num_neuron)
A_bnd = atv(ptb @ W + b)

# 3. Initial Condition Matrix 1 (Displacement: u(x,0)=0)
# Shape: (N_ini, num_neuron)
A_ini1 = atv(pt_ini @ W + b)

# 4. Initial Condition Matrix 2 (Velocity: du/dt(x,0)=0)
# Shape: (N_ini, num_neuron)
A_ini2 = ini_residual(W, b, pt_ini)

# Penalty weights for the loss function (Least Squares weighting)
p_pde = 1
p_bnd = 10
p_ini = 10

# Assemble the Global Matrix A by stacking vertically
# Total Rows: N_pde + 2*N_bnd + 2*N_ini
# Total Columns: num_neuron
A = torch.cat([p_pde*A_pde, p_bnd*A_bnd, p_ini*A_ini1, p_ini*A_ini2], dim=0)

# ========== Construct Target Vector (The "F" Vector) ==========
# PDE Target: Source Term f(x,t)
F_pde = torch.zeros_like(pti[:, 0:1])

# Define Source Term 1: Gaussian modulated cosine
source_term1 = 2 * torch.cos(omega * pti[:, 0:1]) * torch.exp(-((pti[:, 1:2] - 1.5)**2) / (2*sigma**2))
# Define Source Term 2: Higher frequency Gaussian modulated cosine
source_term2 = 5 * torch.cos(2 * omega * pti[:, 0:1]) * torch.exp(-((pti[:, 1:2] - 4.0)**2) / (2*sigma**2))

F_pde += source_term1 + source_term2

# Boundary and Initial Targets (Homogeneous conditions = 0)
F_bnd = torch.zeros_like(ptb[:, 0:1])
F_ini1 = torch.zeros_like(pt_ini[:, 0:1])
F_ini2 = torch.zeros_like(pt_ini[:, 0:1])

# Assemble Global Target Vector F
F = torch.cat([p_pde*F_pde, p_bnd*F_bnd, p_ini*F_ini1, p_ini*F_ini2], dim=0)

# ========== Least Squares Solver ==========
# Solve the overdetermined system A * Ur = F for Ur
# torch.linalg.lstsq computes the minimum-norm solution.
print("Solving linear system...")
Ur = torch.linalg.lstsq(A, F)[0]

end_time = time.time()
print('Total solving time: ', end_time-start_time)

# ========== Prediction and Visualization ==========
start_time = time.time()

# Create a grid for testing/visualization
t_test = torch.linspace(0, T, 101, device=device)
x_test = torch.linspace(0, L, 101, device=device)
T_mesh, X = torch.meshgrid(t_test, x_test)

# Flatten grid for batch processing
test_points = torch.cat((T_mesh.reshape(-1, 1), X.reshape(-1, 1)), dim=1)

# Predict Solution: u(x,t) = sum( Ur_i * tanh(W_i * [t,x] + b_i) )
Z_test = test_points @ W + b
u_pred = (atv(Z_test) @ Ur).reshape(X.shape)

end_time = time.time()
print('Prediction time: ', end_time-start_time)

# Load and prepare analytical solution for comparison (if available)
try:
    u_analytical = np.load(DATA_DIR / '1d_fdtd_two_sources_cos_1e-4.npy')
    # Downsample to match the prediction grid size (101x101) assuming original is finer
    u_analytical = u_analytical[0::5,0::5]
    print("Analytical solution loaded.")
except FileNotFoundError:
    print("Analytical solution file not found. Skipping comparison.")

print('Prediction Finished')

# ========== Visualization ==========
plt.figure(figsize=(18, 6.6))

# 1. Predicted Solution Plot
plt.subplot(1, 3, 1)
# Create a pseudocolor plot for the RFNN prediction
im_pred = plt.pcolormesh(X.cpu().numpy(), T_mesh.cpu().numpy(), u_pred.cpu().detach().numpy(), cmap='jet', shading='auto')
cbar_pred = plt.colorbar(im_pred, orientation='vertical', pad=0.02)
# Set colorbar label and position
cbar_pred.ax.set_title('E(V/m)', fontweight='bold', fontsize=12)
plt.xlabel('x', fontweight='bold', fontsize=14)
plt.ylabel('t', fontweight='bold', fontsize=14)
plt.title('RFNN Solution', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
# Add subplot annotation (a)
plt.annotate('(a)', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

# 2. Analytical Solution Plot
plt.subplot(1, 3, 2)
# Create a pseudocolor plot for the analytical solution
im_analytical = plt.pcolormesh(X.cpu().numpy(), T_mesh.cpu().numpy(), u_analytical, cmap='jet', shading='auto')
cbar_analytical = plt.colorbar(im_analytical, orientation='vertical', pad=0.02)
# Set colorbar label and position
cbar_analytical.ax.set_title('E(V/m)', fontweight='bold', fontsize=12)
plt.xlabel('x', fontweight='bold', fontsize=14)
plt.ylabel('t', fontweight='bold', fontsize=14)
plt.title('Analytical Solution', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
# Add subplot annotation (b)
plt.annotate('(b)', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

# 3. Error Plot
plt.subplot(1, 3, 3)
# Calculate the absolute error: Analytical - Predicted
error = u_analytical - u_pred.cpu().detach().numpy()
# Use 'coolwarm' colormap to visualize positive and negative deviations
im_error = plt.pcolormesh(X.cpu().numpy(), T_mesh.cpu().numpy(), error, cmap='coolwarm', shading='auto')
cbar_error = plt.colorbar(im_error, orientation='vertical', pad=0.02)

# Configure colorbar for scientific notation to handle small error values
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # Force scientific notation
cbar_error.ax.yaxis.set_major_formatter(formatter)

# Adjust the position of the exponent text (10^n) for better aesthetics
cbar_error.ax.yaxis.offsetText.set_fontsize(12)
cbar_error.ax.yaxis.offsetText.set_position((2.3, 1.1))  # Move the offset text up and right

# Set colorbar title with increased padding for spacing
cbar_error.ax.set_title('E(V/m)', fontweight='bold', fontsize=12, pad=20)

plt.xlabel('x', fontweight='bold', fontsize=14)
plt.ylabel('t', fontweight='bold', fontsize=14)
plt.title('Error', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
# Add subplot annotation (c)
plt.annotate('(c)', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout()

# Add a main title to the figure
plt.suptitle('Field Distribution and Errors from RFNN', fontsize=20, fontweight='bold')
# Re-apply tight layout to accommodate the main title
plt.tight_layout()
# Save the figure as a high-quality PDF
plt.savefig('1D_sigma_0.1_rfnn_two_cos_sources.pdf', format='pdf', bbox_inches='tight')

# ========== Error Calculation ==========
# Calculate Relative L2 Error
# Formula: ||u_pred - u_exact|| / ||u_exact||
rel_err = np.linalg.norm(u_pred.cpu().numpy() - u_analytical) / np.linalg.norm(u_analytical)
print(f"Relative L2 Error: {rel_err:.2%}")

# Display the plot
plt.show()
