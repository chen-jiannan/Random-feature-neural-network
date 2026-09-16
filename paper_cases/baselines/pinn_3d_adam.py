"""Paper reproduction script for the Adam-only PINN baseline.

This file is a cleaned, English-only version of the corresponding code supplied with
the manuscript. GPU indices are not hard-coded; set CUDA_VISIBLE_DEVICES externally
if a specific GPU is desired.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

L, T = 1, 1
sigma = 0.2

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        x = self.linears[-1](x)
        return x

    def net_u(self, t, x, y, z):
        u = self.forward(torch.cat([t, x, y, z], dim=1))
        return u

    def net_u_t(self, t, x, y, z):
        u = self.net_u(t, x, y, z)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        return u_t

    def rec_wave(self, t, x, y, z):
        source_1 = torch.cos(2 * torch.pi * t) * torch.exp(
            -((x - 0.1)**2 + (y - 0.6)**2 + (z - 0.1)**2) / (2 * sigma**2)
        )

        source_2 = torch.cos(4 * torch.pi * t) * torch.exp(
            -((x - 0.7)**2 + (y - 0.2)**2 + (z - 0.4)**2) / (2 * sigma**2)
        )

        time_gaussian = torch.exp(-((t - 0.2)**2) / (2 * 0.2**2))
        space_gaussian = torch.exp(
            -((x - 0.4)**2 + (y - 0.8)**2 + (z - 0.9)**2) / (2 * 0.3**2)
        )
        source_3 = time_gaussian * space_gaussian

        return source_1 + source_2 + source_3

    def net_f(self, t, x, y, z, c=0.5):
        u = self.net_u(t, x, y, z)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

        f = u_tt - c**2 * (u_xx + u_yy + u_zz) - self.rec_wave(t, x, y, z)
        return f

layers = [4] + 4*[32] + [1]
net = PINN(layers)

def generate_data(N0, Nb, Nf):
    t0 = torch.zeros((N0, 1), requires_grad=True).to(device)
    xyz0 = torch.rand(N0, 3).to(device) * L
    X0 = torch.cat([t0, xyz0[:, 0:1], xyz0[:, 1:2], xyz0[:, 2:3]], dim=1)

    tb = torch.rand(Nb, 1).view(-1, 1).to(device) * T
    xyzb0 = torch.rand(Nb, 3).to(device) * L
    xyzb0[:, 0] = 0
    xyzb1 = torch.rand(Nb, 3).to(device) * L
    xyzb1[:, 0] = L
    xyzb2 = torch.rand(Nb, 3).to(device) * L
    xyzb2[:, 1] = 0
    xyzb3 = torch.rand(Nb, 3).to(device) * L
    xyzb3[:, 1] = L
    xyzb4 = torch.rand(Nb, 3).to(device) * L
    xyzb4[:, 2] = 0
    xyzb5 = torch.rand(Nb, 3).to(device) * L
    xyzb5[:, 2] = L

    Xb0 = torch.cat([tb, xyzb0[:, 0:1], xyzb0[:, 1:2], xyzb0[:, 2:3]], dim=1)
    Xb1 = torch.cat([tb, xyzb1[:, 0:1], xyzb1[:, 1:2], xyzb1[:, 2:3]], dim=1)
    Xb2 = torch.cat([tb, xyzb2[:, 0:1], xyzb2[:, 1:2], xyzb2[:, 2:3]], dim=1)
    Xb3 = torch.cat([tb, xyzb3[:, 0:1], xyzb3[:, 1:2], xyzb3[:, 2:3]], dim=1)
    Xb4 = torch.cat([tb, xyzb4[:, 0:1], xyzb4[:, 1:2], xyzb4[:, 2:3]], dim=1)
    Xb5 = torch.cat([tb, xyzb5[:, 0:1], xyzb5[:, 1:2], xyzb5[:, 2:3]], dim=1)

    xyzf = torch.rand((Nf, 3), requires_grad=True).to(device) * L
    tf = torch.rand((Nf, 1), requires_grad=True).to(device) * T
    Xf = torch.cat([tf, xyzf[:, 0:1], xyzf[:, 1:2], xyzf[:, 2:3]], dim=1)

    return X0, Xb0, Xb1, Xb2, Xb3, Xb4, Xb5, Xf

def compute_loss(net, X0, Xb0, Xb1, Xb2, Xb3, Xb4, Xb5, Xf):
    u0_pred = net.net_u(X0[:, 0:1], X0[:, 1:2], X0[:, 2:3], X0[:, 3:4])
    u0t_pred = net.net_u_t(X0[:, 0:1], X0[:, 1:2], X0[:, 2:3], X0[:, 3:4])
    u_b0_pred = net.net_u(Xb0[:, 0:1], Xb0[:, 1:2], Xb0[:, 2:3], Xb0[:, 3:4])
    u_b1_pred = net.net_u(Xb1[:, 0:1], Xb1[:, 1:2], Xb1[:, 2:3], Xb1[:, 3:4])
    u_b2_pred = net.net_u(Xb2[:, 0:1], Xb2[:, 1:2], Xb2[:, 2:3], Xb2[:, 3:4])
    u_b3_pred = net.net_u(Xb3[:, 0:1], Xb3[:, 1:2], Xb3[:, 2:3], Xb3[:, 3:4])
    u_b4_pred = net.net_u(Xb4[:, 0:1], Xb4[:, 1:2], Xb4[:, 2:3], Xb4[:, 3:4])
    u_b5_pred = net.net_u(Xb5[:, 0:1], Xb5[:, 1:2], Xb5[:, 2:3], Xb5[:, 3:4])
    f_pred = net.net_f(Xf[:, 0:1], Xf[:, 1:2], Xf[:, 2:3], Xf[:, 3:4])

    loss_u0 = torch.mean((u0_pred)**2)
    loss_u0t = torch.mean((u0t_pred)**2)
    loss_ub0 = torch.mean((u_b0_pred)**2)
    loss_ub1 = torch.mean((u_b1_pred)**2)
    loss_ub2 = torch.mean((u_b2_pred)**2)
    loss_ub3 = torch.mean((u_b3_pred)**2)
    loss_ub4 = torch.mean((u_b4_pred)**2)
    loss_ub5 = torch.mean((u_b5_pred)**2)
    loss_f = torch.mean(f_pred**2)

    loss = (
        10*loss_u0 + 10*loss_u0t +
        10*loss_ub0 + 10*loss_ub1 + 10*loss_ub2 +
        10*loss_ub3 + 10*loss_ub4 + 10*loss_ub5 +
        loss_f
    )

    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)

N0 = 10000
Nb = 20000
Nf = 40000

optimizer = torch.optim.Adam(net.parameters())
epochs = 500000

start_time = time.time()

best_loss = float('inf')
loss_history = []
best_model_state = None
X0, Xb0, Xb1, Xb2, Xb3, Xb4, Xb5, Xf = generate_data(N0, Nb, Nf)

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    loss = compute_loss(net, X0, Xb0, Xb1, Xb2, Xb3, Xb4, Xb5, Xf)

    loss_history.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = net.state_dict()

    if epoch % 20000 == 0:
        end_time = time.time()
        cost_time = end_time - start_time
        print(f'Epoch {epoch}, Loss: {loss.item()}, Trainging Time: {cost_time:.3f}')

    loss.backward(retain_graph=True)
    optimizer.step()

end_time = time.time()
print('Training time : ', end_time - start_time)

torch.save(best_model_state, f'3D_best_model_{sigma}_{epochs}_three_sources.pth')

file_name = f"3D_Loss_{sigma}_{epochs}_three_sources.pkl"

with open(file_name, 'wb') as file:
    pickle.dump(loss_history, file)

plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_history)), loss_history, label='Loss', color='blue')

plt.yscale('log')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log10(Loss)', fontsize=12)
plt.title('Loss Curve (Log Scale)', fontsize=14)
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()

net.load_state_dict(best_model_state)
net.eval()
prediction_start = time.time()

nt, nx, ny, nz = 151, 101, 101, 101
t_axis = np.linspace(0.0, 1.0, nt)
x_axis = np.linspace(0.0, 1.0, nx)
y_axis = np.linspace(0.0, 1.0, ny)
z_axis = np.linspace(0.0, 1.0, nz)
Tg, Xg, Yg, Zg = np.meshgrid(t_axis, x_axis, y_axis, z_axis, indexing="ij")
flat = np.stack([Tg.ravel(), Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

u_flat = np.empty(flat.shape[0], dtype=np.float32)
with torch.no_grad():
    for i in range(0, flat.shape[0], 10000):
        pts = torch.tensor(flat[i:i+10000], dtype=torch.float32, device=device)
        u_flat[i:i+10000] = net(pts).squeeze(1).cpu().numpy()

u_pred = u_flat.reshape(nt, nx, ny, nz)
print("Prediction time:", time.time() - prediction_start)
reference = np.load(DATA_DIR / "3d_fdtd_three_sources_2.5e-3.npy")
rel_err = np.linalg.norm(u_pred - reference) / np.linalg.norm(reference)
print(f"Relative L2 error: {rel_err:.2%}")
