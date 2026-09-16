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

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

L, T = 5, 5
sigma = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def net_u(self, t, x):
        return self.forward(torch.cat([t, x], dim=1))

    def net_u_t(self, t, x):
        u = self.net_u(t, x)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        return u_t

    def rec_wave(self, t, x):
        return (
            2 * torch.cos(2 * torch.pi * t) * torch.exp(-((x - 1.5) ** 2) / (2 * sigma ** 2))
            + 5 * torch.cos(4 * torch.pi * t) * torch.exp(-((x - 4.0) ** 2) / (2 * sigma ** 2))
        )

    def net_f(self, t, x, c=1.0):
        u = self.net_u(t, x)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        f = u_tt - c**2 * u_xx - self.rec_wave(t, x)
        return f

layers = [2] + 4 * [32] + [1]
net = PINN(layers).to(device)

def generate_data(N0, Nb, Nf):
    t0 = torch.zeros((N0, 1), device=device)
    x0 = torch.rand(N0, 1, device=device) * L
    X0 = torch.cat([t0, x0], dim=1)

    tb = torch.rand(Nb, 1, device=device) * T
    xb0 = torch.zeros((Nb, 1), device=device)
    xb1 = torch.ones((Nb, 1), device=device) * L

    Xb0 = torch.cat([tb, xb0], dim=1)
    Xb1 = torch.cat([tb, xb1], dim=1)

    xf = torch.rand((Nf, 1), device=device) * L
    tf = torch.rand((Nf, 1), device=device) * T
    Xf = torch.cat([tf, xf], dim=1)

    return X0, Xb0, Xb1, Xf

def compute_loss(net, X0, Xb0, Xb1, Xf):
    X0_ = X0.detach().clone().requires_grad_(True)
    Xf_ = Xf.detach().clone().requires_grad_(True)
    Xb0_ = Xb0.detach().clone()
    Xb1_ = Xb1.detach().clone()

    u0_pred = net.net_u(X0_[:, 0:1], X0_[:, 1:2])
    u0t_pred = net.net_u_t(X0_[:, 0:1], X0_[:, 1:2])
    u_b0_pred = net.net_u(Xb0_[:, 0:1], Xb0_[:, 1:2])
    u_b1_pred = net.net_u(Xb1_[:, 0:1], Xb1_[:, 1:2])
    f_pred = net.net_f(Xf_[:, 0:1], Xf_[:, 1:2])

    loss_u0 = torch.mean(u0_pred ** 2)
    loss_u0t = torch.mean(u0t_pred ** 2)
    loss_ub0 = torch.mean(u_b0_pred ** 2)
    loss_ub1 = torch.mean(u_b1_pred ** 2)
    loss_f = torch.mean(f_pred ** 2)

    loss = loss_u0 + loss_u0t + loss_ub0 + loss_ub1 + loss_f
    return loss

N0 = 10000
Nb = 20000
Nf = 20000
X0, Xb0, Xb1, Xf = generate_data(N0, Nb, Nf)

# =========================
# =========================
adam_epochs = 200000

optimizer_adam = torch.optim.Adam(net.parameters(), lr=1e-3)


start_time = time.time()
loss_history = []

print("Stage 1: Adam training...")
for epoch in range(adam_epochs+1):
    optimizer_adam.zero_grad()
    loss = compute_loss(net, X0, Xb0, Xb1, Xf)
    loss.backward()
    optimizer_adam.step()

    loss_history.append(loss.item())

    if epoch % 10000 == 0:
        cost_time = time.time() - start_time
        print(f'[Adam] Epoch {epoch}, Loss: {loss.item():.8e}, Training Time: {cost_time:.3f}s')

# =========================
# =========================

end_time = time.time()
print("Adam training time:", end_time - start_time)

MODEL_FILE = REPO_ROOT / "outputs" / "pinn_1d_adam.pth"
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), MODEL_FILE)

net.eval()
prediction_start = time.time()
x_test = torch.linspace(0, L, 101, device=device)
t_test = torch.linspace(0, T, 101, device=device)
T_mesh, X_mesh = torch.meshgrid(t_test, x_test, indexing="ij")
test_points = torch.cat((T_mesh.reshape(-1, 1), X_mesh.reshape(-1, 1)), dim=1)
with torch.no_grad():
    u_pred = net(test_points).reshape(T_mesh.shape).cpu().numpy()
print("Prediction time:", time.time() - prediction_start)

reference = np.load(DATA_DIR / "1d_fdtd_two_sources_cos_1e-4.npy")
reference = reference[::5, ::5]
rel_err = np.linalg.norm(u_pred - reference) / np.linalg.norm(reference)
print(f"Relative L2 error: {rel_err:.2%}")
