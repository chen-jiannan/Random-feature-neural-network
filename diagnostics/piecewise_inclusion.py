from __future__ import annotations
import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfnn_experiment_core import (
    rel_l2, resolve_device, resolve_dtype, save_csv_rows,
    set_seed, sync, tanh_d1, tanh_d2
)


def parse_args():
    p = argparse.ArgumentParser(
        description="2D piecewise heterogeneous circular-inclusion stress test and contrast sweep."
    )
    p.add_argument("--output", default="results_2d_inclusion_v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    p.add_argument("--mode", choices=["single", "sweep"], default="sweep")
    p.add_argument("--run-partitioned", action="store_true")
    p.add_argument("--solver", choices=["ridge", "lstsq"], default="ridge")
    p.add_argument("--ridge-rel", type=float, default=1e-10)
    p.add_argument("--q-in", type=float, default=0.25)
    p.add_argument(
        "--q-in-values", nargs="+", type=float, default=None,
        help="Custom inclusion q=c^2 values for sweep."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Random seeds for the contrast sweep."
    )
    p.add_argument(
        "--plot-q-in", type=float, default=0.25,
        help="Representative contrast for t=0.5 field/error plots."
    )
    return p.parse_args()


class RF2D:
    def __init__(self, W, b):
        self.W = W
        self.b = b


def sample_uniform_disk(n, radius, device, dtype):
    theta = 2.0 * math.pi * torch.rand((n, 1), device=device, dtype=dtype)
    rr = radius * torch.sqrt(torch.rand((n, 1), device=device, dtype=dtype))
    return rr * torch.cos(theta), rr * torch.sin(theta)


def sample_uniform_annulus(n, r0, r1, device, dtype):
    theta = 2.0 * math.pi * torch.rand((n, 1), device=device, dtype=dtype)
    rr = torch.sqrt(
        r0 * r0 + (r1 * r1 - r0 * r0) *
        torch.rand((n, 1), device=device, dtype=dtype)
    )
    return rr * torch.cos(theta), rr * torch.sin(theta)


def make_rf_2d(M, T_end, outer_radius, inclusion_radius, weight_range,
               seed, device, dtype, region="global"):
    set_seed(seed)
    W = (2.0 * torch.rand((3, M), device=device, dtype=dtype) - 1.0) * weight_range
    tc = torch.rand((1, M), device=device, dtype=dtype) * T_end

    if region == "global":
        xc, yc = sample_uniform_disk(M, outer_radius, device, dtype)
    elif region == "inside":
        xc, yc = sample_uniform_disk(M, inclusion_radius, device, dtype)
    elif region == "outside":
        xc, yc = sample_uniform_annulus(
            M, inclusion_radius, outer_radius, device, dtype
        )
    else:
        raise ValueError(region)

    centers = torch.cat([tc, xc.T, yc.T], dim=0)
    b = -torch.sum(centers * W, dim=0, keepdim=True)
    return RF2D(W, b)


def phi(rf, pts):
    return torch.tanh(pts @ rf.W + rf.b)


def phi_t(rf, pts):
    z = pts @ rf.W + rf.b
    return rf.W[0:1, :] * tanh_d1(z)


def phi_tt(rf, pts):
    z = pts @ rf.W + rf.b
    return (rf.W[0:1, :] ** 2) * tanh_d2(z)


def phi_x(rf, pts):
    z = pts @ rf.W + rf.b
    return rf.W[1:2, :] * tanh_d1(z)


def phi_y(rf, pts):
    z = pts @ rf.W + rf.b
    return rf.W[2:3, :] * tanh_d1(z)


def phi_lap(rf, pts):
    z = pts @ rf.W + rf.b
    return (rf.W[1:2, :] ** 2 + rf.W[2:3, :] ** 2) * tanh_d2(z)


def exact_spatial_torch(pts, outer_radius, inclusion_radius, q_out, q_in):
    x, y = pts[:, 1:2], pts[:, 2:3]
    r2 = x * x + y * y
    a2 = inclusion_radius ** 2
    beta = -q_out / q_in
    alpha = outer_radius ** 2 - a2 + (q_out / q_in) * a2
    psi_out = outer_radius ** 2 - r2
    psi_in = alpha + beta * r2
    return torch.where(r2 < a2, psi_in, psi_out)


def exact_solution_torch(pts, outer_radius, inclusion_radius, q_out, q_in, omega):
    t = pts[:, 0:1]
    psi = exact_spatial_torch(
        pts, outer_radius, inclusion_radius, q_out, q_in
    )
    return psi * (1.0 - torch.cos(omega * t))


def exact_source_torch(pts, outer_radius, inclusion_radius, q_out, q_in, omega):
    t = pts[:, 0:1]
    psi = exact_spatial_torch(
        pts, outer_radius, inclusion_radius, q_out, q_in
    )
    g = 1.0 - torch.cos(omega * t)
    return (omega ** 2) * torch.cos(omega * t) * psi + 4.0 * q_out * g


def sample_problem_points(n_pde, n_bnd, n_ini, n_interface,
                          T_end, outer_radius, inclusion_radius,
                          seed, device, dtype):
    set_seed(seed)

    xp, yp = sample_uniform_disk(n_pde, outer_radius, device, dtype)
    tp = torch.rand((n_pde, 1), device=device, dtype=dtype) * T_end
    pde = torch.cat([tp, xp, yp], dim=1)

    th = 2.0 * math.pi * torch.rand((n_bnd, 1), device=device, dtype=dtype)
    tb = torch.rand((n_bnd, 1), device=device, dtype=dtype) * T_end
    bnd = torch.cat([
        tb,
        outer_radius * torch.cos(th),
        outer_radius * torch.sin(th),
    ], dim=1)

    xi, yi = sample_uniform_disk(n_ini, outer_radius, device, dtype)
    ini = torch.cat([torch.zeros_like(xi), xi, yi], dim=1)

    thi = 2.0 * math.pi * torch.rand((n_interface, 1), device=device, dtype=dtype)
    ti = torch.rand((n_interface, 1), device=device, dtype=dtype) * T_end
    interface = torch.cat([
        ti,
        inclusion_radius * torch.cos(thi),
        inclusion_radius * torch.sin(thi),
    ], dim=1)
    normals = torch.cat([torch.cos(thi), torch.sin(thi)], dim=1)

    return pde, bnd, ini, interface, normals


def sample_test_points(n_test, T_end, outer_radius, seed, device, dtype):
    set_seed(seed)
    x, y = sample_uniform_disk(n_test, outer_radius, device, dtype)
    t = torch.rand((n_test, 1), device=device, dtype=dtype) * T_end
    return torch.cat([t, x, y], dim=1)


def q_at_points(pts, inclusion_radius, q_out, q_in):
    r2 = pts[:, 1:2] ** 2 + pts[:, 2:3] ** 2
    return torch.where(
        r2 < inclusion_radius ** 2,
        torch.full_like(r2, q_in),
        torch.full_like(r2, q_out),
    )


def normal_derivative(rf, pts, normals):
    return normals[:, 0:1] * phi_x(rf, pts) + normals[:, 1:2] * phi_y(rf, pts)


def solve_system(A, F, solver, ridge_rel):
    sync(A.device)
    tic = time.perf_counter()

    if solver == "lstsq":
        beta = torch.linalg.lstsq(A, F).solution
    elif solver == "ridge":
        G = A.T @ A
        rhs = A.T @ F
        scale = torch.trace(G) / G.shape[0]
        I = torch.eye(G.shape[0], device=A.device, dtype=A.dtype)
        beta = torch.linalg.solve(G + ridge_rel * scale * I, rhs)
    else:
        raise ValueError(solver)

    sync(A.device)
    return beta, time.perf_counter() - tic


def fit_global(M, weight_range, pde, bnd, ini, interface, normals,
               T_end, outer_radius, inclusion_radius, q_out, q_in, omega,
               seed, device, dtype, solver, ridge_rel, interface_weight=10.0):
    rf = make_rf_2d(
        M, T_end, outer_radius, inclusion_radius, weight_range,
        seed, device, dtype, region="global"
    )
    q = q_at_points(pde, inclusion_radius, q_out, q_in)
    A_pde = phi_tt(rf, pde) - q * phi_lap(rf, pde)
    A_bnd = 10.0 * phi(rf, bnd)
    A_ini_u = 10.0 * phi(rf, ini)
    A_ini_v = 10.0 * phi_t(rf, ini)

    # A single globally smooth representation sees the same normal derivative
    # from both sides. Flux continuity therefore becomes a stringent constraint.
    gn = normal_derivative(rf, interface, normals)
    A_flux = interface_weight * (q_out - q_in) * gn

    A = torch.cat(
        [A_pde, A_bnd, A_ini_u, A_ini_v, A_flux], dim=0
    )
    F = torch.cat([
        exact_source_torch(
            pde, outer_radius, inclusion_radius, q_out, q_in, omega
        ),
        torch.zeros((bnd.shape[0], 1), device=device, dtype=dtype),
        torch.zeros((ini.shape[0], 1), device=device, dtype=dtype),
        torch.zeros((ini.shape[0], 1), device=device, dtype=dtype),
        torch.zeros((interface.shape[0], 1), device=device, dtype=dtype),
    ], dim=0)

    beta, solve_time = solve_system(A, F, solver, ridge_rel)
    return rf, beta, solve_time


def fit_partitioned(M_each, weight_range, pde, bnd, ini, interface, normals,
                    T_end, outer_radius, inclusion_radius, q_out, q_in, omega,
                    seed, device, dtype, solver, ridge_rel,
                    interface_weight=10.0):
    rf_out = make_rf_2d(
        M_each, T_end, outer_radius, inclusion_radius, weight_range,
        seed, device, dtype, region="outside"
    )
    rf_in = make_rf_2d(
        M_each, T_end, outer_radius, inclusion_radius, weight_range,
        seed + 1, device, dtype, region="inside"
    )

    def inside_mask(pts):
        return pts[:, 1] ** 2 + pts[:, 2] ** 2 < inclusion_radius ** 2

    mp = inside_mask(pde)
    mi = inside_mask(ini)
    p_out, p_in = pde[~mp], pde[mp]
    i_out, i_in = ini[~mi], ini[mi]

    M = M_each
    zpo = torch.zeros((p_out.shape[0], M), device=device, dtype=dtype)
    zpi = torch.zeros((p_in.shape[0], M), device=device, dtype=dtype)
    zb = torch.zeros((bnd.shape[0], M), device=device, dtype=dtype)
    zio = torch.zeros((i_out.shape[0], M), device=device, dtype=dtype)
    zii = torch.zeros((i_in.shape[0], M), device=device, dtype=dtype)

    A_p_out = torch.cat([
        phi_tt(rf_out, p_out) - q_out * phi_lap(rf_out, p_out), zpo
    ], dim=1)
    A_p_in = torch.cat([
        zpi, phi_tt(rf_in, p_in) - q_in * phi_lap(rf_in, p_in)
    ], dim=1)
    A_bnd = torch.cat([10.0 * phi(rf_out, bnd), 10.0 * zb], dim=1)
    A_iou = torch.cat([10.0 * phi(rf_out, i_out), 10.0 * zio], dim=1)
    A_iov = torch.cat([10.0 * phi_t(rf_out, i_out), 10.0 * zio], dim=1)
    A_iiu = torch.cat([10.0 * zii, 10.0 * phi(rf_in, i_in)], dim=1)
    A_iiv = torch.cat([10.0 * zii, 10.0 * phi_t(rf_in, i_in)], dim=1)

    A_cont = interface_weight * torch.cat([
        phi(rf_out, interface), -phi(rf_in, interface)
    ], dim=1)
    gn_out = normal_derivative(rf_out, interface, normals)
    gn_in = normal_derivative(rf_in, interface, normals)
    A_flux = interface_weight * torch.cat([
        q_out * gn_out, -q_in * gn_in
    ], dim=1)

    A = torch.cat([
        A_p_out, A_p_in, A_bnd,
        A_iou, A_iov, A_iiu, A_iiv,
        A_cont, A_flux,
    ], dim=0)

    zeros = lambda n: torch.zeros((n, 1), device=device, dtype=dtype)
    F = torch.cat([
        exact_source_torch(
            p_out, outer_radius, inclusion_radius, q_out, q_in, omega
        ),
        exact_source_torch(
            p_in, outer_radius, inclusion_radius, q_out, q_in, omega
        ),
        zeros(bnd.shape[0]),
        zeros(i_out.shape[0]), zeros(i_out.shape[0]),
        zeros(i_in.shape[0]), zeros(i_in.shape[0]),
        zeros(interface.shape[0]), zeros(interface.shape[0]),
    ], dim=0)

    beta, solve_time = solve_system(A, F, solver, ridge_rel)
    return rf_out, rf_in, beta[:M_each], beta[M_each:], solve_time


@torch.no_grad()
def predict_global_points(rf, beta, pts, batch=65536):
    out = torch.empty((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    for i0 in range(0, pts.shape[0], batch):
        i1 = min(i0 + batch, pts.shape[0])
        out[i0:i1] = phi(rf, pts[i0:i1]) @ beta
    return out


@torch.no_grad()
def predict_partitioned_points(rf_out, rf_in, beta_out, beta_in,
                               pts, inclusion_radius, batch=65536):
    r2 = pts[:, 1] ** 2 + pts[:, 2] ** 2
    inside = r2 < inclusion_radius ** 2
    out = torch.empty((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)

    for mask, rf, beta in [
        (~inside, rf_out, beta_out),
        (inside, rf_in, beta_in),
    ]:
        ids = torch.where(mask)[0]
        for i0 in range(0, ids.numel(), batch):
            sel = ids[i0:i0 + batch]
            out[sel] = phi(rf, pts[sel]) @ beta
    return out


def relative_l2_torch(pred, exact):
    return float(
        (torch.linalg.norm(pred - exact) / torch.linalg.norm(exact))
        .detach().cpu()
    )


@torch.no_grad()
def field_at_time_global(rf, beta, t_plot, grid, outer_radius, device, dtype):
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    mask = X * X + Y * Y <= outer_radius ** 2
    pts_np = np.stack([
        np.full(mask.sum(), t_plot),
        X[mask],
        Y[mask],
    ], axis=1)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)
    vals = predict_global_points(rf, beta, pts).squeeze(1).cpu().numpy()
    field = np.full_like(X, np.nan, dtype=np.float64)
    field[mask] = vals
    return field, mask


@torch.no_grad()
def field_at_time_partitioned(rf_out, rf_in, beta_out, beta_in,
                              t_plot, grid, outer_radius, inclusion_radius,
                              device, dtype):
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    mask = X * X + Y * Y <= outer_radius ** 2
    pts_np = np.stack([
        np.full(mask.sum(), t_plot),
        X[mask],
        Y[mask],
    ], axis=1)
    pts = torch.tensor(pts_np, device=device, dtype=dtype)
    vals = predict_partitioned_points(
        rf_out, rf_in, beta_out, beta_in, pts, inclusion_radius
    ).squeeze(1).cpu().numpy()
    field = np.full_like(X, np.nan, dtype=np.float64)
    field[mask] = vals
    return field, mask


def exact_field_numpy(t_plot, grid, outer_radius, inclusion_radius,
                      q_out, q_in, omega):
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    r2 = X * X + Y * Y
    a2 = inclusion_radius ** 2
    beta = -q_out / q_in
    alpha = outer_radius ** 2 - a2 + (q_out / q_in) * a2
    psi = np.where(
        r2 < a2,
        alpha + beta * r2,
        outer_radius ** 2 - r2,
    )
    U = psi * (1.0 - np.cos(omega * t_plot))
    mask = r2 <= outer_radius ** 2
    U = np.where(mask, U, np.nan)
    return U, mask


def plot_field(field, grid, title, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(
        field.T, origin="lower",
        extent=[grid[0], grid[-1], grid[0], grid[-1]],
        aspect="equal"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    outer_radius = 1.0
    inclusion_radius = 0.35
    q_out = 1.0
    T_end = 1.0
    omega = 2.0 * math.pi
    t_plot = 0.5  # maximum-amplitude time; unlike t=1, the exact field is not zero.

    if args.preset == "smoke":
        M = 80
        n_pde, n_bnd, n_ini, n_interface = 500, 160, 160, 160
        n_test = 3000
        grid_n = 61
        weight_range = 5.0
        default_q = [1.0, 0.25]
        default_seeds = [5]
    else:
        M = 2000
        n_pde, n_bnd, n_ini, n_interface = 20000, 6000, 8000, 6000
        n_test = 100000
        grid_n = 151
        weight_range = 5.0
        default_q = [1.0, 0.64, 0.36, 0.25, 0.16]
        default_seeds = [5, 17, 29]

    q_values = (
        [args.q_in] if args.mode == "single"
        else (args.q_in_values if args.q_in_values is not None else default_q)
    )
    seeds = args.seeds if args.seeds is not None else default_seeds

    if any(q <= 0 for q in q_values):
        raise ValueError("All q_in values must be positive.")

    # Fixed independent test set for every contrast and seed.
    test_pts = sample_test_points(
        n_test, T_end, outer_radius, seed=9999,
        device=device, dtype=dtype
    )

    rows = []
    representative = {}

    for q_in in q_values:
        speed_ratio = math.sqrt(q_in / q_out)

        for seed in seeds:
            # Same collocation realization for this seed at every contrast.
            pde, bnd, ini, interface, normals = sample_problem_points(
                n_pde, n_bnd, n_ini, n_interface,
                T_end, outer_radius, inclusion_radius,
                seed=3000 + seed, device=device, dtype=dtype
            )

            exact_test = exact_solution_torch(
                test_pts, outer_radius, inclusion_radius,
                q_out, q_in, omega
            )

            print(
                f"q_in={q_in:.4f} (c_in/c_out={speed_ratio:.3f}), "
                f"seed={seed}: global RF"
            )
            rf, beta, solve_time = fit_global(
                M, weight_range, pde, bnd, ini, interface, normals,
                T_end, outer_radius, inclusion_radius, q_out, q_in, omega,
                seed=seed, device=device, dtype=dtype,
                solver=args.solver, ridge_rel=args.ridge_rel
            )
            pred = predict_global_points(rf, beta, test_pts)
            err = relative_l2_torch(pred, exact_test)

            rows.append({
                "method": "global_RF",
                "q_out": q_out,
                "q_in": q_in,
                "c_in_over_c_out": speed_ratio,
                "seed": seed,
                "features_total": M,
                "N_test": n_test,
                "solver": args.solver,
                "relative_L2_testset": err,
                "solve_time_s_diagnostic": solve_time,
            })
            print(f"  global_RF RelL2(test) = {err:.6e}")

            if abs(q_in - args.plot_q_in) < 1e-12 and seed == seeds[0]:
                representative["global"] = (rf, beta, q_in)

            if args.run_partitioned:
                print("  partitioned/interface RF")
                M_each = M // 2
                rfo, rfi, bo, bi, solve_time_p = fit_partitioned(
                    M_each, weight_range, pde, bnd, ini, interface, normals,
                    T_end, outer_radius, inclusion_radius, q_out, q_in, omega,
                    seed=1000 + seed, device=device, dtype=dtype,
                    solver=args.solver, ridge_rel=args.ridge_rel
                )
                pred_p = predict_partitioned_points(
                    rfo, rfi, bo, bi, test_pts, inclusion_radius
                )
                err_p = relative_l2_torch(pred_p, exact_test)

                rows.append({
                    "method": "partitioned_interface_RF",
                    "q_out": q_out,
                    "q_in": q_in,
                    "c_in_over_c_out": speed_ratio,
                    "seed": seed,
                    "features_total": 2 * M_each,
                    "N_test": n_test,
                    "solver": args.solver,
                    "relative_L2_testset": err_p,
                    "solve_time_s_diagnostic": solve_time_p,
                })
                print(f"  partitioned RelL2(test) = {err_p:.6e}")

                if abs(q_in - args.plot_q_in) < 1e-12 and seed == seeds[0]:
                    representative["partitioned"] = (rfo, rfi, bo, bi, q_in)

            del pde, bnd, ini, interface, normals, exact_test, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_csv_rows(rows, out / "inclusion_contrast_sweep_v2.csv")

    # Aggregate mean/std over seeds.
    summary = []
    methods = sorted(set(r["method"] for r in rows))
    for method in methods:
        for q_in in q_values:
            vals = np.array([
                r["relative_L2_testset"] for r in rows
                if r["method"] == method and abs(r["q_in"] - q_in) < 1e-14
            ], dtype=float)
            if len(vals) == 0:
                continue
            summary.append({
                "method": method,
                "q_in": q_in,
                "c_in_over_c_out": math.sqrt(q_in / q_out),
                "relative_L2_mean": float(vals.mean()),
                "relative_L2_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "relative_L2_min": float(vals.min()),
                "relative_L2_max": float(vals.max()),
                "n_seeds": int(len(vals)),
            })
    save_csv_rows(summary, out / "inclusion_contrast_summary_v2.csv")

    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    for method in methods:
        rr = [r for r in summary if r["method"] == method]
        rr.sort(key=lambda z: z["c_in_over_c_out"], reverse=True)
        ax.errorbar(
            [r["c_in_over_c_out"] for r in rr],
            [r["relative_L2_mean"] for r in rr],
            yerr=[r["relative_L2_std"] for r in rr],
            marker="o", capsize=4,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("Wave-speed ratio c_in / c_out")
    ax.set_ylabel("Relative L2 error on independent test set")
    ax.set_yscale("log")
    ax.set_title("Piecewise-inclusion contrast sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "inclusion_error_vs_contrast_v2.pdf")
    plt.close(fig)

    # Representative field/error plots at t=0.5, where the exact amplitude is maximal.
    grid = np.linspace(-outer_radius, outer_radius, grid_n)
    plot_q = args.plot_q_in
    if plot_q not in q_values:
        print(
            f"Representative plot q_in={plot_q} was not included in this run; "
            "no field plots were generated."
        )
    else:
        exact_field, mask = exact_field_numpy(
            t_plot, grid, outer_radius, inclusion_radius,
            q_out, plot_q, omega
        )
        plot_field(
            exact_field, grid,
            f"Exact field at t={t_plot:g}, q_in={plot_q:g}",
            out / "exact_field_t0p5_v2.pdf"
        )

        if "global" in representative:
            rf, beta, q_in = representative["global"]
            field, _ = field_at_time_global(
                rf, beta, t_plot, grid, outer_radius, device, dtype
            )
            plot_field(
                field, grid,
                f"Global RF field at t={t_plot:g}",
                out / "global_RF_field_t0p5_v2.pdf"
            )
            err_field = np.where(mask, field - exact_field, np.nan)
            plot_field(
                err_field, grid,
                f"Global RF error at t={t_plot:g}",
                out / "global_RF_error_t0p5_v2.pdf"
            )

        if "partitioned" in representative:
            rfo, rfi, bo, bi, q_in = representative["partitioned"]
            field_p, _ = field_at_time_partitioned(
                rfo, rfi, bo, bi, t_plot, grid,
                outer_radius, inclusion_radius, device, dtype
            )
            plot_field(
                field_p, grid,
                f"Interface-aware RF field at t={t_plot:g}",
                out / "partitioned_RF_field_t0p5_v2.pdf"
            )
            err_field_p = np.where(mask, field_p - exact_field, np.nan)
            plot_field(
                err_field_p, grid,
                f"Interface-aware RF error at t={t_plot:g}",
                out / "partitioned_RF_error_t0p5_v2.pdf"
            )

    print(f"Saved results to: {out.resolve()}")
    print(
        "The sweep uses the same total number of random features for global and "
        "partitioned models, and reports mean/std across seeds."
    )
    print(
        "The t=0.5 plots replace the previous t=1 final-time plots, because the "
        "manufactured exact field is identically zero at t=1."
    )


if __name__ == "__main__":
    main()
