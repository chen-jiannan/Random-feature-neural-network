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

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c = 0.5
L = 1.0
T = 1.0
num_neuron = 10000
r_neuron = [2, 4, 4, 4]
d = 4
sigma = 0.2

atv = lambda x: torch.tanh(x)
diff1_atv = lambda x: 1 - torch.tanh(x)**2
diff2_atv = lambda x: 2 * torch.tanh(x) * (torch.tanh(x)**2 - 1)

def generate_points():
    N_pde = 10000
    N_bnd = 10000
    N_ini = 10000

    ti = torch.rand(N_pde, 1, device=device) * T
    xi = torch.rand(N_pde, 1, device=device) * L
    yi = torch.rand(N_pde, 1, device=device) * L
    zi = torch.rand(N_pde, 1, device=device) * L
    pti = torch.cat((ti, xi, yi, zi), dim=1)

    tb = torch.rand(N_bnd, 1, device=device) * T
    xb = torch.rand(N_bnd, 1, device=device) * L
    yb = torch.rand(N_bnd, 1, device=device) * L
    zb = torch.rand(N_bnd, 1, device=device) * L

    ptb_x0 = torch.cat((tb, torch.zeros_like(xb), yb, zb), dim=1)
    ptb_xL = torch.cat((tb, torch.full_like(xb, L), yb, zb), dim=1)
    ptb_y0 = torch.cat((tb, xb, torch.zeros_like(yb), zb), dim=1)
    ptb_yL = torch.cat((tb, xb, torch.full_like(yb, L), zb), dim=1)
    ptb_z0 = torch.cat((tb, xb, yb, torch.zeros_like(zb)), dim=1)
    ptb_zL = torch.cat((tb, xb, yb, torch.full_like(zb, L)), dim=1)

    ptb = torch.cat((ptb_x0, ptb_xL, ptb_y0, ptb_yL, ptb_z0, ptb_zL), dim=0)

    ti_ini = torch.zeros(N_ini, 1, device=device)
    xi_ini = torch.rand(N_ini, 1, device=device) * L
    yi_ini = torch.rand(N_ini, 1, device=device) * L
    zi_ini = torch.rand(N_ini, 1, device=device) * L
    pt_ini = torch.cat((ti_ini, xi_ini, yi_ini, zi_ini), dim=1)

    return pti, ptb, pt_ini

def wave_equation_residual(W, b, points, c):
    T = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]
    Z = points[:, 3:4]

    W_t = W[0:1, :]
    W_x = W[1:2, :]
    W_y = W[2:3, :]
    W_z = W[3:4, :]

    Z_hidden = T@W_t + X@W_x + Y@W_y + Z@W_z + b

    phi_tt = torch.sum(W_t**2, 0) * diff2_atv(Z_hidden)
    phi_xx = torch.sum(W_x**2, 0) * diff2_atv(Z_hidden)
    phi_yy = torch.sum(W_y**2, 0) * diff2_atv(Z_hidden)
    phi_zz = torch.sum(W_z**2, 0) * diff2_atv(Z_hidden)

    return phi_tt - (c**2)*(phi_xx + phi_yy + phi_zz)

def ini_residual(W, b, points):
    T = points[:, 0:1]
    X = points[:, 1:2]
    Y = points[:, 2:3]
    Z = points[:, 3:4]

    W_t = W[0:1, :]
    W_x = W[1:2, :]
    W_y = W[2:3, :]
    W_z = W[3:4, :]

    Z_hidden = T@W_t + X@W_x + Y@W_y + Z@W_z + b
    phi_t = W_t * (1 - atv(Z_hidden)**2)

    return phi_t

start_time = time.time()
W1 = torch.tensor(np.random.uniform(-r_neuron[0], r_neuron[0], (1, num_neuron)), dtype=torch.float32)
W2 = torch.tensor(np.random.uniform(-r_neuron[1], r_neuron[1], (1, num_neuron)), dtype=torch.float32)
W3 = torch.tensor(np.random.uniform(-r_neuron[2], r_neuron[2], (1, num_neuron)), dtype=torch.float32)
W4 = torch.tensor(np.random.uniform(-r_neuron[3], r_neuron[3], (1, num_neuron)), dtype=torch.float32)
W = torch.cat((W1, W2, W3, W4), dim=0).to(device)

b1 = torch.tensor(np.random.uniform(0, T, (1, num_neuron)), dtype=torch.float32)
b2 = torch.tensor(np.random.uniform(0, L, (1, num_neuron)), dtype=torch.float32)
b3 = torch.tensor(np.random.uniform(0, L, (1, num_neuron)), dtype=torch.float32)
b4 = torch.tensor(np.random.uniform(0, L, (1, num_neuron)), dtype=torch.float32)
b = torch.cat((b1, b2, b3, b4), dim=0).to(device)
b = -torch.sum(b * W, dim=0)

pti, ptb, pt_ini = generate_points()

A_pde = wave_equation_residual(W, b, pti, c)
A_bnd = atv(ptb @ W + b)
A_ini1 = atv(pt_ini @ W + b)
A_ini2 = ini_residual(W, b, pt_ini)

p_pde = 1
p_bnd = 10
p_ini = 10

A = torch.cat([p_pde*A_pde, p_bnd*A_bnd, p_ini*A_ini1, p_ini*A_ini2], dim=0)

F_pde = torch.zeros_like(pti[:, 0:1])

source_1 = torch.cos(2 * np.pi * pti[:, 0:1]) * torch.exp(
    -((pti[:, 1:2] - 0.1)**2 + (pti[:, 2:3] - 0.6)**2 + (pti[:, 3:4] - 0.1)**2) / (2 * sigma**2)
)

source_2 = torch.cos(4 * np.pi * pti[:, 0:1]) * torch.exp(
    -((pti[:, 1:2] - 0.7)**2 + (pti[:, 2:3] - 0.2)**2 + (pti[:, 3:4] - 0.4)**2) / (2 * sigma**2)
)

time_gaussian = torch.exp(-((pti[:, 0:1] - 0.2)**2) / (2 * 0.2**2))
space_gaussian = torch.exp(
    -((pti[:, 1:2] - 0.4)**2 + (pti[:, 2:3] - 0.8)**2 + (pti[:, 3:4] - 0.9)**2) / (2 * 0.3**2)
)
source_3 = time_gaussian * space_gaussian

source_term = source_1 + source_2 + source_3
F_pde += source_term

F_bnd = torch.zeros_like(ptb[:, 0:1])
F_ini1 = torch.zeros_like(pt_ini[:, 0:1])
F_ini2 = torch.zeros_like(pt_ini[:, 0:1])

F = torch.cat([p_pde*F_pde, p_bnd*F_bnd, p_ini*F_ini1, p_ini*F_ini2], dim=0)

Ur = torch.linalg.lstsq(A, F)[0]

end_time = time.time()
print('Training time: ', end_time-start_time)


start_time = time.time()
nt = 151
nx = 101
ny = 101
nz = 101

t_test = torch.linspace(0, T, nt, device=device)
x_test = torch.linspace(0, L, nx, device=device)
y_test = torch.linspace(0, L, ny, device=device)
z_test = torch.linspace(0, L, nz, device=device)

T_mesh, X, Y, Z = torch.meshgrid(t_test, x_test, y_test, z_test, indexing='ij')
test_points = torch.cat((T_mesh.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), dim=1).to(device)

batch_size = 30000
u_pred = []

with torch.no_grad():
    for i in range(0, test_points.shape[0], batch_size):
        batch = test_points[i:i+batch_size]
        hidden = torch.tanh(batch @ W + b)  # (batch_size, num_neuron)
        output = hidden @ Ur                # (batch_size, 1)
        u_pred.append(output.cpu())

end_time = time.time()
print('Prediction time: ', end_time-start_time)

u_pred = torch.cat(u_pred).numpy().reshape(T_mesh.shape)
print('Prediction finished.')


fdtd = np.load(DATA_DIR / '3d_fdtd_three_sources_2.5e-3.npy')
error = u_pred - fdtd
rel_err_rfnn_2 = np.linalg.norm(u_pred - fdtd) / np.linalg.norm(fdtd)
print(f"Relative L2 Error: {rel_err_rfnn_2:.2%}")
