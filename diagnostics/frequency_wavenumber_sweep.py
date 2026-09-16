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
        description="Extended frequency/wavenumber sweep for the global RFNN."
    )
    p.add_argument("--output", default="results_frequency_wavenumber_v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    p.add_argument("--weight-range", type=float, default=5.0)
    p.add_argument("--solver", choices=["lstsq", "ridge"], default="lstsq")
    p.add_argument("--ridge-rel", type=float, default=1e-10)
    p.add_argument(
        "--ks", nargs="+", type=int, default=None,
        help="Optional custom mode indices, e.g. --ks 16 20 24 32 40 48"
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional custom random seeds."
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
    L, c, T = 1.0, 1.0, 2.0

    if args.preset == "smoke":
        default_ks = [8, 16]
        default_seeds = [5]
        M, n_pde, n_bnd_total, n_ini = 120, 500, 200, 200
    else:
        # This is intentionally the extension of the original k=1,2,4,8,12 sweep.
        default_ks = [16, 20, 24, 32, 40, 48]
        default_seeds = [5, 17, 29]
        M, n_pde, n_bnd_total, n_ini = 5000, 30000, 20000, 10000

    ks = args.ks if args.ks is not None else default_ks
    seeds = args.seeds if args.seeds is not None else default_seeds
    if any(k <= 0 for k in ks):
        raise ValueError("All mode indices k must be positive.")

    # Keep the evaluation grid fine enough for the highest requested k.
    kmax = max(ks)
    if args.preset == "smoke":
        nx = max(81, 4 * kmax + 1)
        nt = max(121, 8 * kmax + 1)
    else:
        # ~16 samples per spatial wavelength and temporal period at kmax.
        nx = max(301, 8 * kmax + 1)
        nt = max(401, 16 * kmax + 1)

    gpu_warmup(device, dtype)

    rows = []
    for k in ks:
        for seed in seeds:
            rf = make_rf_1d(
                M=M, L=L, T=T,
                weight_range_t=args.weight_range,
                weight_range_x=args.weight_range,
                seed=seed, device=device, dtype=dtype,
            )
            pde, bnd, ini = sample_1d_points(
                L, T, n_pde, n_bnd_total, n_ini,
                seed=1000 + seed, device=device, dtype=dtype,
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

            u_exact = exact_grid_1d(tg, xg, k, L, c)
            err = rel_l2(u, u_exact)

            rows.append({
                "k": k,
                "wavelengths_in_domain": k / 2.0,
                "temporal_cycles": c * k * T / (2.0 * L),
                "seed": seed,
                "features": M,
                "N_pde": n_pde,
                "N_bnd_total": n_bnd_total,
                "N_ini": n_ini,
                "weight_range_t": args.weight_range,
                "weight_range_x": args.weight_range,
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
                f"k={k:2d}, seed={seed:2d}, RelL2={err:.6e}, "
                f"solve={solve_time:.3f}s"
            )

            del A, F, beta, pde, bnd, ini
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_csv_rows(rows, out / "frequency_wavenumber_sweep_v2.csv")

    # Paper-oriented summary.
    summary = []
    for k in ks:
        vals = np.array(
            [r["relative_L2"] for r in rows if r["k"] == k], dtype=float
        )
        summary.append({
            "k": k,
            "relative_L2_mean": float(vals.mean()),
            "relative_L2_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "relative_L2_min": float(vals.min()),
            "relative_L2_max": float(vals.max()),
            "n_seeds": int(len(vals)),
        })
    save_csv_rows(summary, out / "frequency_wavenumber_summary_v2.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.errorbar(
        [r["k"] for r in summary],
        [r["relative_L2_mean"] for r in summary],
        yerr=[r["relative_L2_std"] for r in summary],
        marker="o", capsize=4,
    )
    ax.set_xlabel("Spatial mode index k")
    ax.set_ylabel("Relative L2 error")
    ax.set_yscale("log")
    ax.set_title("Extended frequency / wavenumber sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "frequency_error_vs_k_v2.pdf")
    plt.close(fig)

    print(f"Saved results to: {out.resolve()}")
    print(
        "Note: timings are diagnostic only. Use the error/threshold trend as the "
        "primary result unless a separate repeated timing protocol is performed."
    )


if __name__ == "__main__":
    main()
