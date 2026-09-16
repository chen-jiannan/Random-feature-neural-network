from __future__ import annotations
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfnn_experiment_core import (
    assemble_mode_system, exact_grid_1d, make_rf_1d, predict_1d,
    rel_l2, resolve_device, resolve_dtype, sample_1d_points,
    save_csv_rows, solve_linear_system, sync
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extended long-time propagation study for the global RFNN."
    )
    p.add_argument("--output", default="results_long_time_v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    p.add_argument("--weight-range", type=float, default=5.0)
    p.add_argument("--solver", choices=["lstsq", "ridge"], default="lstsq")
    p.add_argument("--ridge-rel", type=float, default=1e-10)
    p.add_argument(
        "--protocol",
        choices=["fixed_budget", "fixed_density", "both"],
        default="fixed_budget",
        help="Run fixed_budget now. fixed_density is intended only for selected T values."
    )
    p.add_argument(
        "--T-values", nargs="+", type=float, default=None,
        help="Optional custom final times, e.g. --T-values 16 32 64"
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional custom random seeds."
    )
    p.add_argument(
        "--allow-large-density", action="store_true",
        help="Allow fixed-density systems whose raw A matrix is estimated above 12 GiB."
    )
    return p.parse_args()


def gpu_warmup(device, dtype):
    if device.type != "cuda":
        return
    x = torch.randn((256, 256), device=device, dtype=dtype)
    for _ in range(3):
        _ = x @ x
    sync(device)


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    # k=2 with L=c=1 has temporal period exactly 1.
    L, c, k = 1.0, 1.0, 2

    if args.preset == "smoke":
        default_T = [4.0, 8.0]
        default_seeds = [5]
        M = 120
        fixed = dict(n_pde=500, n_bnd_total=200, n_ini=200)
        density = dict(n_pde_per_T=350, n_bnd_per_T=150, n_ini=200)
        nx = 81
    else:
        # This extends the original T=1,2,4,8 study.
        default_T = [16.0, 32.0, 64.0]
        default_seeds = [5, 17, 29]
        M = 5000
        fixed = dict(n_pde=30000, n_bnd_total=20000, n_ini=10000)
        density = dict(n_pde_per_T=6000, n_bnd_per_T=4000, n_ini=10000)
        nx = 301

    T_list = args.T_values if args.T_values is not None else default_T
    seeds = args.seeds if args.seeds is not None else default_seeds
    if any(T <= 0 for T in T_list):
        raise ValueError("All final times T must be positive.")

    protocols = (
        ["fixed_budget", "fixed_density"]
        if args.protocol == "both"
        else [args.protocol]
    )

    gpu_warmup(device, dtype)
    rows = []

    bytes_per = 8 if dtype == torch.float64 else 4

    for protocol in protocols:
        for T in T_list:
            if protocol == "fixed_budget":
                n_pde = fixed["n_pde"]
                n_bnd_total = fixed["n_bnd_total"]
                n_ini = fixed["n_ini"]
            else:
                n_pde = int(round(density["n_pde_per_T"] * T))
                n_bnd_total = int(round(density["n_bnd_per_T"] * T))
                n_ini = density["n_ini"]

                n_rows = n_pde + n_bnd_total + 2 * n_ini
                raw_gib = n_rows * M * bytes_per / (1024 ** 3)
                if raw_gib > 12.0 and not args.allow_large_density:
                    print(
                        f"SKIP fixed_density T={T:g}: raw A estimate is "
                        f"{raw_gib:.1f} GiB. Re-run this selected case with "
                        f"--allow-large-density only if your GPU memory is sufficient."
                    )
                    continue

            # About 20 evaluation points per temporal period.
            nt = max(201, int(round(20 * T)) + 1)

            for seed in seeds:
                rf = make_rf_1d(
                    M=M, L=L, T=T,
                    weight_range_t=args.weight_range,
                    weight_range_x=args.weight_range,
                    seed=seed, device=device, dtype=dtype,
                )
                pde, bnd, ini = sample_1d_points(
                    L, T, n_pde, n_bnd_total, n_ini,
                    seed=2000 + seed, device=device, dtype=dtype,
                )

                sync(device)
                tic = time.perf_counter()
                A, F = assemble_mode_system(
                    rf, pde, bnd, ini, k=k, L=L, c=c
                )
                sync(device)
                assembly_time = time.perf_counter() - tic

                beta, solve_time = solve_linear_system(
                    A, F, args.solver, args.ridge_rel
                )

                sync(device)
                tic = time.perf_counter()
                tg, xg, u = predict_1d(
                    rf, beta, L, T, nx=nx, nt=nt
                )
                sync(device)
                prediction_time = time.perf_counter() - tic

                exact = exact_grid_1d(tg, xg, k, L, c)
                err = rel_l2(u, exact)

                rows.append({
                    "protocol": protocol,
                    "T": T,
                    "temporal_periods": T,
                    "seed": seed,
                    "features": M,
                    "N_pde": n_pde,
                    "N_bnd_total": n_bnd_total,
                    "N_ini": n_ini,
                    "weight_range": args.weight_range,
                    "solver": args.solver,
                    "dtype": args.dtype,
                    "eval_nx": nx,
                    "eval_nt": nt,
                    "relative_L2": err,
                    "assembly_time_s_diagnostic": assembly_time,
                    "solve_time_s_diagnostic": solve_time,
                    "prediction_time_s_diagnostic": prediction_time,
                })
                print(
                    f"{protocol:13s} T={T:5.1f}, seed={seed:2d}, "
                    f"RelL2={err:.6e}"
                )

                del A, F, beta, pde, bnd, ini
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    save_csv_rows(rows, out / "long_time_propagation_v2.csv")

    summary = []
    for protocol in protocols:
        for T in T_list:
            vals = np.array([
                r["relative_L2"] for r in rows
                if r["protocol"] == protocol and r["T"] == T
            ], dtype=float)
            if len(vals) == 0:
                continue
            summary.append({
                "protocol": protocol,
                "T": T,
                "relative_L2_mean": float(vals.mean()),
                "relative_L2_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "relative_L2_min": float(vals.min()),
                "relative_L2_max": float(vals.max()),
                "n_seeds": int(len(vals)),
            })
    save_csv_rows(summary, out / "long_time_summary_v2.csv")

    if summary:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for protocol in sorted(set(r["protocol"] for r in summary)):
            rr = [r for r in summary if r["protocol"] == protocol]
            rr.sort(key=lambda z: z["T"])
            ax.errorbar(
                [r["T"] for r in rr],
                [r["relative_L2_mean"] for r in rr],
                yerr=[r["relative_L2_std"] for r in rr],
                marker="o", capsize=4,
                label=protocol.replace("_", " "),
            )
        ax.set_xlabel("Final time T (number of temporal periods)")
        ax.set_ylabel("Relative L2 error")
        ax.set_yscale("log")
        ax.set_title("Extended long-time propagation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "long_time_error_v2.pdf")
        plt.close(fig)

    print(f"Saved results to: {out.resolve()}")
    print(
        "Run fixed_budget first. Only after a clear degradation threshold is found "
        "should fixed_density be repeated at one or two selected T values."
    )


if __name__ == "__main__":
    main()
