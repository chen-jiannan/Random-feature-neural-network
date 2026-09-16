# Global random-feature collocation for time-domain wave equations

This repository contains the cleaned code accompanying the manuscript

**“Global random-feature collocation for time-domain wave equations:
a systematic assessment of resolution, conditioning, and applicability limits.”**

The release focuses on the final paper cases. Exploratory single-source, source-free,
nonlinear, stale plotting, and L-BFGS comparison cells from the development notebooks
have been removed. All comments and user-facing text are in English, and GPU indices
are no longer hard-coded.

## Repository layout

```text
paper_cases/
  rfnn/       Final global-RF/RFNN paper cases
  baselines/  FDTD reference generators and Adam-only PINN baselines
diagnostics/  Frequency, long-time, conditioning, and interface studies
data/         Generated large reference arrays (not committed)
paper_results/ Compact CSV summaries of the reported trends
scripts/      Convenience shell scripts
```

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch should be installed with the CUDA build appropriate for your system if GPU
execution is desired. The scripts automatically use CUDA when available unless a
device is explicitly specified.

## Baseline run order

The RFNN and PINN comparison scripts load FDTD reference arrays from `data/`.
Generate the homogeneous references first:

```bash
python paper_cases/baselines/fdtd_1d_homogeneous.py
python paper_cases/baselines/fdtd_2d_homogeneous.py
python paper_cases/baselines/fdtd_3d_homogeneous.py
```

The 3D reference is computationally and memory intensive.

Then run the global RF cases:

```bash
python paper_cases/rfnn/rfnn_1d_homogeneous.py
python paper_cases/rfnn/rfnn_2d_homogeneous.py
python paper_cases/rfnn/rfnn_3d_homogeneous.py
```

Additional RFNN cases:

```bash
python paper_cases/rfnn/rfnn_1d_damping.py
python paper_cases/rfnn/rfnn_1d_heterogeneous.py
python paper_cases/rfnn/rfnn_2d_damping.py
```

The 2D damping script expects the large reference array
`data/2d_fdtd_ref_2e-4.npy`. That array is not bundled in this source archive because
of its size. See `data/README.md`.

Adam-only PINN baselines are in `paper_cases/baselines/`:

```bash
python paper_cases/baselines/pinn_1d_adam.py
python paper_cases/baselines/pinn_2d_adam.py
python paper_cases/baselines/pinn_3d_adam.py
```

These training runs are expensive and are included for reproducibility rather than
as quick examples.

## Diagnostic experiments

Run smoke tests first:

```bash
python diagnostics/conditioning_solver_study.py --preset smoke
python diagnostics/frequency_wavenumber_sweep.py --preset smoke
python diagnostics/long_time_propagation.py --preset smoke
python diagnostics/piecewise_inclusion.py --preset smoke --mode sweep --run-partitioned
```

Paper-scale runs:

```bash
python diagnostics/conditioning_solver_study.py --preset paper --device cuda
python diagnostics/frequency_wavenumber_sweep.py --preset paper --device cuda
python diagnostics/long_time_propagation.py --preset paper --device cuda --protocol fixed_budget
python diagnostics/piecewise_inclusion.py --preset paper --device cuda --mode sweep --run-partitioned
```

For the interface solver-independence check:

```bash
python diagnostics/piecewise_inclusion.py \
  --preset paper --device cuda --mode single --q-in 0.25 \
  --run-partitioned --solver lstsq --seeds 5 \
  --output results_2d_inclusion_lstsq_check
```

The paper also uses a retuned high-frequency extension:

```bash
python diagnostics/frequency_wavenumber_sweep.py \
  --preset paper --device cuda --ks 28 32 36 --weight-range 10 \
  --output results_frequency_wr10_extension
```

## Choosing a GPU

No script contains a hard-coded GPU number. Use the environment when needed:

```bash
CUDA_VISIBLE_DEVICES=0 python paper_cases/rfnn/rfnn_3d_homogeneous.py
```

## Results and reproducibility

`paper_results/` contains compact CSV summaries used to check the main trends.
Large `.npy` reference arrays, trained `.pth` models, and newly generated result
directories are intentionally excluded from version control.

Timings depend on hardware, CUDA/PyTorch versions, warm-up, synchronization, and dense
linear-algebra backends. The paper's scientific conclusions rely primarily on accuracy,
conditioning, and robustness trends rather than on exact reproduction of every wall time.

Please read `REPRODUCIBILITY_NOTES.md` before creating a permanent release tag.

## Terminology

The manuscript uses **global RF collocation** for the general formulation and **RFNN**
as shorthand for the specific implementation in the baseline experiments. The
interface-aware two-region solver in `diagnostics/piecewise_inclusion.py` is a diagnostic
comparator, not the original global RFNN method.

## Citation

A `CITATION.cff` file is provided for GitHub's citation interface. Update the journal,
DOI, and publication year after acceptance.

## License

No software license is included automatically in this archive. Choose and add the
license appropriate for your group before making the repository public.
