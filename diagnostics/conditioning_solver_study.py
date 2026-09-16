from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfnn_experiment_core import (
    RF1D, assemble_mode_system, exact_grid_1d, make_rf_1d, predict_1d,
    rel_l2, resolve_device, resolve_dtype, sample_1d_points,
    save_csv_rows, solve_linear_system, sync
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Conditioning and linear-solver study with SVD-based kappa(A)."
    )
    p.add_argument("--output", default="results_conditioning_v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    p.add_argument("--weight-range", type=float, default=5.0)
    p.add_argument(
        "--M-values", nargs="+", type=int, default=None,
        help="Optional custom feature counts."
    )
    return p.parse_args()


def gpu_warmup(device):
    if device.type != "cuda":
        return
    x = torch.randn((256, 256), device=device, dtype=torch.float64)
    for _ in range(3):
        _ = x @ x
    sync(device)


def direct_svd_condition(A64):
    """Compute kappa_2(A) directly from singular values in float64.

    This avoids the kappa(A^T A) eigenvalue pathology that produced spurious
    negative minimum eigenvalues / inf values in the previous script.
    """
    device = A64.device
    sync(device)
    tic = time.perf_counter()
    try:
        s = torch.linalg.svdvals(A64)
    except RuntimeError as exc:
        if device.type != "cpu":
            print(
                "GPU SVD failed; falling back to CPU float64 SVD. "
                f"Reason: {str(exc).splitlines()[0]}"
            )
            s = torch.linalg.svdvals(A64.cpu())
        else:
            raise
    if s.device.type == "cuda":
        sync(s.device)
    elapsed = time.perf_counter() - tic

    smax = float(s[0].detach().cpu())
    smin = float(s[-1].detach().cpu())
    cond = float(smax / smin) if smin > 0 else float("inf")

    eps = np.finfo(np.float64).eps
    tol = max(A64.shape) * eps * smax
    rank = int((s.detach().cpu().numpy() > tol).sum())

    return {
        "sigma_max_svd64": smax,
        "sigma_min_svd64": smin,
        "condition_A_svd64": cond,
        "log10_condition_A_svd64": math.log10(cond) if np.isfinite(cond) and cond > 0 else float("inf"),
        "condition_AtA_theoretical": cond * cond if np.isfinite(cond) else float("inf"),
        "numerical_rank_svd64": rank,
        "svd_condition_time_s": elapsed,
    }


def cast_problem(rf64, pde64, bnd64, ini64, dtype):
    """Cast ONE float64 realization to another precision.

    This guarantees that float32 and float64 use identical hidden parameters,
    centers, and collocation locations rather than separately generated RNG streams.
    """
    rf = RF1D(W=rf64.W.to(dtype=dtype), b=rf64.b.to(dtype=dtype))
    return (
        rf,
        pde64.to(dtype=dtype),
        bnd64.to(dtype=dtype),
        ini64.to(dtype=dtype),
    )


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    L, c, T, k = 1.0, 1.0, 2.0, 4

    if args.preset == "smoke":
        default_M = [40, 80, 120]
        n_pde, n_bnd_total, n_ini = 300, 120, 120
        dtype_names = ["float32", "float64"]
        solvers = [("lstsq", None), ("normal", None), ("ridge", 1e-10)]
        nx, nt = 61, 81
    else:
        default_M = [100, 200, 400, 800, 1200]
        n_pde, n_bnd_total, n_ini = 6000, 3000, 3000
        dtype_names = ["float32", "float64"]
        solvers = [
            ("lstsq", None),
            ("qr", None),
            ("normal", None),
            ("ridge", 1e-12),
            ("ridge", 1e-8),
        ]
        nx, nt = 201, 301

    M_list = args.M_values if args.M_values is not None else default_M
    gpu_warmup(device)

    rows = []

    for M in M_list:
        # Generate the random realization ONCE in float64.
        rf64_base = make_rf_1d(
            M=M, L=L, T=T,
            weight_range_t=args.weight_range,
            weight_range_x=args.weight_range,
            seed=5, device=device, dtype=torch.float64,
        )
        pde64_base, bnd64_base, ini64_base = sample_1d_points(
            L, T, n_pde, n_bnd_total, n_ini,
            seed=1005, device=device, dtype=torch.float64,
        )

        # Assemble the reference float64 matrix once and compute kappa(A) from its SVD.
        A64_ref, F64_ref = assemble_mode_system(
            rf64_base, pde64_base, bnd64_base, ini64_base,
            k=k, L=L, c=c
        )
        cond_info = direct_svd_condition(A64_ref)

        print(
            f"M={M:4d}: kappa_2(A)={cond_info['condition_A_svd64']:.6e}, "
            f"rank={cond_info['numerical_rank_svd64']}/{M}"
        )

        for dtype_name in dtype_names:
            dtype = resolve_dtype(dtype_name)

            if dtype == torch.float64:
                rf = rf64_base
                pde, bnd, ini = pde64_base, bnd64_base, ini64_base
                A, F = A64_ref, F64_ref
            else:
                rf, pde, bnd, ini = cast_problem(
                    rf64_base, pde64_base, bnd64_base, ini64_base, dtype
                )
                A, F = assemble_mode_system(
                    rf, pde, bnd, ini, k=k, L=L, c=c
                )

            for method, ridge_rel in solvers:
                label = method if method != "ridge" else f"ridge_{ridge_rel:.0e}"

                try:
                    beta, solve_time = solve_linear_system(
                        A, F, method=method,
                        ridge_rel=1e-10 if ridge_rel is None else ridge_rel,
                    )
                    tg, xg, u = predict_1d(
                        rf, beta, L, T, nx=nx, nt=nt
                    )
                    exact = exact_grid_1d(tg, xg, k, L, c)
                    err = rel_l2(u, exact)

                    residual = float(
                        (torch.linalg.norm(A @ beta - F) / torch.linalg.norm(F))
                        .detach().cpu()
                    )
                    beta_norm = float(torch.linalg.norm(beta).detach().cpu())
                    status = "ok"
                except RuntimeError as exc:
                    solve_time = float("nan")
                    err = float("nan")
                    residual = float("nan")
                    beta_norm = float("nan")
                    status = "failed:" + str(exc).splitlines()[0][:160]

                row = {
                    "dtype": dtype_name,
                    "features": M,
                    "solver": label,
                    "same_random_realization_across_dtypes": True,
                    **cond_info,
                    "relative_L2": err,
                    "relative_residual": residual,
                    "beta_norm": beta_norm,
                    "solve_time_s_diagnostic": solve_time,
                    "status": status,
                }
                rows.append(row)

                print(
                    f"  dtype={dtype_name:7s} solver={label:13s} "
                    f"RelL2={err:.6e} residual={residual:.3e}"
                )

            if dtype != torch.float64:
                del A, F, pde, bnd, ini

        del A64_ref, F64_ref, pde64_base, bnd64_base, ini64_base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv_rows(rows, out / "conditioning_solver_study_v2.csv")

    # One condition curve only: condition is evaluated on the shared float64 realization.
    cond_rows = []
    for M in M_list:
        rr = [r for r in rows if r["features"] == M]
        if rr:
            cond_rows.append({
                "features": M,
                "condition_A_svd64": rr[0]["condition_A_svd64"],
                "log10_condition_A_svd64": rr[0]["log10_condition_A_svd64"],
                "condition_AtA_theoretical": rr[0]["condition_AtA_theoretical"],
                "sigma_max_svd64": rr[0]["sigma_max_svd64"],
                "sigma_min_svd64": rr[0]["sigma_min_svd64"],
                "numerical_rank_svd64": rr[0]["numerical_rank_svd64"],
            })
    save_csv_rows(cond_rows, out / "conditioning_summary_v2.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        [r["features"] for r in cond_rows],
        [r["condition_A_svd64"] for r in cond_rows],
        marker="o",
    )
    ax.set_xlabel("Number of random features")
    ax.set_ylabel("Condition number kappa_2(A)")
    ax.set_yscale("log")
    ax.set_title("SVD-based conditioning of the RF collocation matrix")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "condition_number_svd_vs_features_v2.pdf")
    plt.close(fig)

    for dtype_name in dtype_names:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        labels = sorted(set(
            r["solver"] for r in rows if r["dtype"] == dtype_name
        ))
        for label in labels:
            rr = [
                r for r in rows
                if r["dtype"] == dtype_name and r["solver"] == label
            ]
            rr.sort(key=lambda z: z["features"])
            ax.plot(
                [r["features"] for r in rr],
                [r["relative_L2"] for r in rr],
                marker="o", label=label,
            )
        ax.set_xlabel("Number of random features")
        ax.set_ylabel("Relative L2 error")
        ax.set_yscale("log")
        ax.set_title(f"Linear-solver sensitivity ({dtype_name})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"solver_error_{dtype_name}_v2.pdf")
        plt.close(fig)

    print(f"Saved results to: {out.resolve()}")
    print(
        "Key change: kappa_2(A) is now computed directly from float64 singular values, "
        "and float32/float64 use exactly the same random realization."
    )


if __name__ == "__main__":
    main()
