"""Shared utilities for the supplementary RFNN assessment experiments.

The implementation follows the manuscript's global random-feature collocation idea:
hidden parameters are sampled once and kept fixed, while only the output
coefficients are obtained from a linear least-squares problem.

Input ordering is explicit throughout:
    1D: [t, x]
    2D: [t, x, y]

This removes the center/weight-order ambiguity that can easily occur in notebooks.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float32", "fp32", "single"}:
        return torch.float32
    if name in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def tanh_d1(z: torch.Tensor) -> torch.Tensor:
    th = torch.tanh(z)
    return 1.0 - th * th


def tanh_d2(z: torch.Tensor) -> torch.Tensor:
    th = torch.tanh(z)
    return 2.0 * th * (th * th - 1.0)


@dataclass
class RF1D:
    W: torch.Tensor  # [2, M], ordering [t, x]
    b: torch.Tensor  # [1, M]


def make_rf_1d(
    M: int,
    L: float,
    T: float,
    weight_range_t: float,
    weight_range_x: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RF1D:
    """Sample global tanh random features centered over the space-time domain."""
    set_seed(seed)
    Wt = (2.0 * torch.rand((1, M), device=device, dtype=dtype) - 1.0) * weight_range_t
    Wx = (2.0 * torch.rand((1, M), device=device, dtype=dtype) - 1.0) * weight_range_x
    W = torch.cat([Wt, Wx], dim=0)

    center_t = torch.rand((1, M), device=device, dtype=dtype) * T
    center_x = torch.rand((1, M), device=device, dtype=dtype) * L
    centers = torch.cat([center_t, center_x], dim=0)
    b = -torch.sum(centers * W, dim=0, keepdim=True)
    return RF1D(W=W, b=b)


def phi_1d(rf: RF1D, pts: torch.Tensor) -> torch.Tensor:
    return torch.tanh(pts @ rf.W + rf.b)


def phi_t_1d(rf: RF1D, pts: torch.Tensor) -> torch.Tensor:
    z = pts @ rf.W + rf.b
    Wt = rf.W[0:1, :]
    return Wt * tanh_d1(z)


def phi_tt_1d(rf: RF1D, pts: torch.Tensor) -> torch.Tensor:
    z = pts @ rf.W + rf.b
    Wt = rf.W[0:1, :]
    return (Wt * Wt) * tanh_d2(z)


def phi_xx_1d(rf: RF1D, pts: torch.Tensor) -> torch.Tensor:
    z = pts @ rf.W + rf.b
    Wx = rf.W[1:2, :]
    return (Wx * Wx) * tanh_d2(z)


def sample_1d_points(
    L: float,
    T: float,
    n_pde: int,
    n_bnd_total: int,
    n_ini: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample points; n_bnd_total is the total across BOTH boundaries."""
    set_seed(seed)
    ti = torch.rand((n_pde, 1), device=device, dtype=dtype) * T
    xi = torch.rand((n_pde, 1), device=device, dtype=dtype) * L
    pde = torch.cat([ti, xi], dim=1)

    n_left = n_bnd_total // 2
    n_right = n_bnd_total - n_left
    tl = torch.rand((n_left, 1), device=device, dtype=dtype) * T
    tr = torch.rand((n_right, 1), device=device, dtype=dtype) * T
    bnd_l = torch.cat([tl, torch.zeros_like(tl)], dim=1)
    bnd_r = torch.cat([tr, torch.full_like(tr, L)], dim=1)
    bnd = torch.cat([bnd_l, bnd_r], dim=0)

    xi0 = torch.rand((n_ini, 1), device=device, dtype=dtype) * L
    ini = torch.cat([torch.zeros_like(xi0), xi0], dim=1)
    return pde, bnd, ini


def mode_omega(k: int, L: float, c: float) -> float:
    return c * k * math.pi / L


def manufactured_mode_exact(pts: torch.Tensor, k: int, L: float, c: float) -> torch.Tensor:
    """u = sin(k*pi*x/L) * [1-cos(omega*t)], omega=c*k*pi/L.

    This satisfies zero Dirichlet data, u(x,0)=0, and u_t(x,0)=0.
    """
    t = pts[:, 0:1]
    x = pts[:, 1:2]
    omega = mode_omega(k, L, c)
    return torch.sin(k * math.pi * x / L) * (1.0 - torch.cos(omega * t))


def manufactured_mode_source(pts: torch.Tensor, k: int, L: float, c: float) -> torch.Tensor:
    """For the manufactured mode, f=u_tt-c^2 u_xx=omega^2 sin(k*pi*x/L)."""
    x = pts[:, 1:2]
    omega = mode_omega(k, L, c)
    return (omega * omega) * torch.sin(k * math.pi * x / L)


def assemble_mode_system(
    rf: RF1D,
    pde: torch.Tensor,
    bnd: torch.Tensor,
    ini: torch.Tensor,
    k: int,
    L: float,
    c: float,
    w_pde: float = 1.0,
    w_bnd: float = 10.0,
    w_ini_u: float = 10.0,
    w_ini_v: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A_pde = phi_tt_1d(rf, pde) - (c * c) * phi_xx_1d(rf, pde)
    A_bnd = phi_1d(rf, bnd)
    A_ini_u = phi_1d(rf, ini)
    A_ini_v = phi_t_1d(rf, ini)

    f_pde = manufactured_mode_source(pde, k, L, c)
    z_bnd = torch.zeros((bnd.shape[0], 1), device=bnd.device, dtype=bnd.dtype)
    z_ini = torch.zeros((ini.shape[0], 1), device=ini.device, dtype=ini.dtype)

    A = torch.cat(
        [w_pde * A_pde, w_bnd * A_bnd, w_ini_u * A_ini_u, w_ini_v * A_ini_v],
        dim=0,
    )
    F = torch.cat(
        [w_pde * f_pde, w_bnd * z_bnd, w_ini_u * z_ini, w_ini_v * z_ini],
        dim=0,
    )
    return A, F


def solve_linear_system(
    A: torch.Tensor,
    F: torch.Tensor,
    method: str = "lstsq",
    ridge_rel: float = 1e-10,
) -> Tuple[torch.Tensor, float]:
    """Solve A beta ~= F and return beta and synchronized solve time."""
    device = A.device
    sync(device)
    tic = time.perf_counter()
    method = method.lower()

    if method == "lstsq":
        beta = torch.linalg.lstsq(A, F).solution
    elif method == "qr":
        Q, R = torch.linalg.qr(A, mode="reduced")
        rhs = Q.transpose(0, 1) @ F
        beta = torch.linalg.solve_triangular(R, rhs, upper=True)
    elif method == "normal":
        G = A.transpose(0, 1) @ A
        rhs = A.transpose(0, 1) @ F
        beta = torch.linalg.solve(G, rhs)
    elif method == "ridge":
        G = A.transpose(0, 1) @ A
        rhs = A.transpose(0, 1) @ F
        scale = torch.trace(G) / G.shape[0]
        lam = ridge_rel * scale
        beta = torch.linalg.solve(
            G + lam * torch.eye(G.shape[0], device=device, dtype=A.dtype),
            rhs,
        )
    else:
        raise ValueError(f"Unknown solver method: {method}")

    sync(device)
    elapsed = time.perf_counter() - tic
    return beta, elapsed


@torch.no_grad()
def predict_1d(
    rf: RF1D,
    beta: torch.Tensor,
    L: float,
    T: float,
    nx: int,
    nt: int,
    batch_size: int = 65536,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device, dtype = rf.W.device, rf.W.dtype
    t = torch.linspace(0.0, T, nt, device=device, dtype=dtype)
    x = torch.linspace(0.0, L, nx, device=device, dtype=dtype)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    pts = torch.cat([tt.reshape(-1, 1), xx.reshape(-1, 1)], dim=1)

    out = torch.empty((pts.shape[0], 1), device=device, dtype=dtype)
    for i0 in range(0, pts.shape[0], batch_size):
        i1 = min(i0 + batch_size, pts.shape[0])
        out[i0:i1] = phi_1d(rf, pts[i0:i1]) @ beta

    return (
        t.detach().cpu().numpy(),
        x.detach().cpu().numpy(),
        out.reshape(nt, nx).detach().cpu().numpy(),
    )


def exact_grid_1d(t: np.ndarray, x: np.ndarray, k: int, L: float, c: float) -> np.ndarray:
    tt, xx = np.meshgrid(t, x, indexing="ij")
    omega = mode_omega(k, L, c)
    return np.sin(k * np.pi * xx / L) * (1.0 - np.cos(omega * tt))


def estimate_condition_from_gram(A: torch.Tensor) -> Tuple[float, float, float]:
    """Estimate kappa_2(A) from eigenvalues of A^T A.

    Returns (cond_A_est, cond_gram, min_eig_gram). If numerical rank loss
    makes the smallest eigenvalue non-positive, the condition estimates are inf.
    """
    G = A.transpose(0, 1) @ A
    G = 0.5 * (G + G.transpose(0, 1))
    eig = torch.linalg.eigvalsh(G)
    lam_min = float(eig[0].detach().cpu())
    lam_max = float(eig[-1].detach().cpu())
    if lam_min <= 0.0 or not np.isfinite(lam_min) or not np.isfinite(lam_max):
        return float("inf"), float("inf"), lam_min
    cond_g = lam_max / lam_min
    return float(math.sqrt(cond_g)), float(cond_g), lam_min


def save_csv_rows(rows, path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
