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

L, T = 5, 1
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

    def net_u(self, t, x, y):
        u = self.forward(torch.cat([t, x, y], dim=1))
        return u

    def net_u_t(self, t, x, y):
        u = self.net_u(t, x, y)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        return u_t

    def rec_wave(self, t, x, y):
        source_1 = torch.cos(2 * torch.pi * t) * torch.exp(-((x - 1.5)**2 + (y - 1.5)**2) / (2 * sigma**2))

        time_gaussian = torch.exp(-((t - 0.1)**2) / (2 * 0.1**2))
        space_gaussian = torch.exp(-((x - 4.0)**2 + (y - 4.0)**2) / (2 * sigma**2))
        source_2 = time_gaussian * space_gaussian

        return source_1 + source_2

    def net_f(self, t, x, y, c=1.0):
        u = self.net_u(t, x, y)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

        f = u_tt - c**2 * (u_xx + u_yy) - self.rec_wave(t, x, y)
        return f

layers = [3] + 4*[32] + [1]
net = PINN(layers)

def generate_data(N0, Nb, Nf):
    t0 = torch.zeros((N0, 1), requires_grad=True).to(device)
    xy0 = torch.rand(N0, 2).to(device) * L
    X0 = torch.cat([t0, xy0[:, 0:1], xy0[:, 1:2]], dim=1)

    tb = torch.rand(Nb, 1).view(-1, 1).to(device) * T
    xyb0 = torch.rand(Nb, 2).to(device) * L
    xyb0[:, 0] = 0
    xyb1 = torch.rand(Nb, 2).to(device) * L
    xyb1[:, 0] = L
    xyb2 = torch.rand(Nb, 2).to(device) * L
    xyb2[:, 1] = 0
    xyb3 = torch.rand(Nb, 2).to(device) * L
    xyb3[:, 1] = L

    Xb0 = torch.cat([tb, xyb0[:, 0:1], xyb0[:, 1:2]], dim=1)
    Xb1 = torch.cat([tb, xyb1[:, 0:1], xyb1[:, 1:2]], dim=1)
    Xb2 = torch.cat([tb, xyb2[:, 0:1], xyb2[:, 1:2]], dim=1)
    Xb3 = torch.cat([tb, xyb3[:, 0:1], xyb3[:, 1:2]], dim=1)

    xyf = torch.rand((Nf, 2), requires_grad=True).to(device) * L
    tf = torch.rand((Nf, 1), requires_grad=True).to(device) * T
    Xf = torch.cat([tf, xyf[:, 0:1], xyf[:, 1:2]], dim=1)

    return X0, Xb0, Xb1, Xb2, Xb3, Xf

def compute_loss(net, X0, Xb0, Xb1, Xb2, Xb3, Xf):
    u0_pred = net.net_u(X0[:, 0:1], X0[:, 1:2], X0[:, 2:3])
    u0t_pred = net.net_u_t(X0[:, 0:1], X0[:, 1:2], X0[:, 2:3])
    u_b0_pred = net.net_u(Xb0[:, 0:1], Xb0[:, 1:2], Xb0[:, 2:3])
    u_b1_pred = net.net_u(Xb1[:, 0:1], Xb1[:, 1:2], Xb1[:, 2:3])
    u_b2_pred = net.net_u(Xb2[:, 0:1], Xb2[:, 1:2], Xb2[:, 2:3])
    u_b3_pred = net.net_u(Xb3[:, 0:1], Xb3[:, 1:2], Xb3[:, 2:3])
    f_pred = net.net_f(Xf[:, 0:1], Xf[:, 1:2], Xf[:, 2:3])

    loss_u0 = torch.mean((u0_pred)**2)
    loss_u0t = torch.mean((u0t_pred)**2)
    loss_ub0 = torch.mean((u_b0_pred)**2)
    loss_ub1 = torch.mean((u_b1_pred)**2)
    loss_ub2 = torch.mean((u_b2_pred)**2)
    loss_ub3 = torch.mean((u_b3_pred)**2)
    loss_f = torch.mean(f_pred**2)

    loss = 10*loss_u0 + 10*loss_u0t + 10*loss_ub0 + 10*loss_ub1 + 10*loss_ub2 + 10*loss_ub3 + loss_f

    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)

N0 = 10000
Nb = 20000
Nf = 20000

optimizer = torch.optim.Adam(net.parameters())
epochs = 300000

start_time = time.time()

best_loss = float('inf')
loss_history = []
best_model_state = None
X0, Xb0, Xb1, Xb2, Xb3, Xf = generate_data(N0, Nb, Nf)

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    loss = compute_loss(net, X0, Xb0, Xb1, Xb2, Xb3, Xf)

    loss_history.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = net.state_dict()

    if epoch % 10000 == 0:
        end_time = time.time()
        cost_time = end_time - start_time
        print(f'Epoch {epoch}, Loss: {loss.item()}, Trainging Time: {cost_time:.3f}')

    loss.backward(retain_graph=True)
    optimizer.step()

end_time = time.time()
print('Training time : ', end_time - start_time)

torch.save(best_model_state, f'2d_best_model_{sigma}_two_sources.pth')

file_name = f"2d_Loss_{sigma}_300000.pkl"

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
nt, nx, ny = 151, 501, 501
t_test = torch.linspace(0, T, nt, device=device)
x_test = torch.linspace(0, L, nx, device=device)
y_test = torch.linspace(0, L, ny, device=device)
T_mesh, X_mesh, Y_mesh = torch.meshgrid(t_test, x_test, y_test, indexing="ij")
test_points = torch.cat(
    (T_mesh.reshape(-1,1), X_mesh.reshape(-1,1), Y_mesh.reshape(-1,1)), dim=1
)
pred = []
with torch.no_grad():
    for i in range(0, test_points.shape[0], 20000):
        pred.append(net(test_points[i:i+20000]).cpu())
u_pred = torch.cat(pred).numpy().reshape(nt, nx, ny)
print("Prediction time:", time.time() - prediction_start)
reference = np.load(DATA_DIR / "2d_fdtd_two_source_2e-3.npy")
rel_err = np.linalg.norm(u_pred - reference) / np.linalg.norm(reference)
print(f"Relative L2 error: {rel_err:.2%}")
