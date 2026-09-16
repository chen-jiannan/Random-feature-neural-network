# Reproducibility notes

## Canonical parameters

- 3D RFNN: `T=1.0`, `M=10000`, temporal weights in `[-2,2]`, spatial weights in
  `[-4,4]`, corrected initial-velocity derivative
  `W_t * (1 - tanh(z)**2)`.
- Final confirmed 3D RFNN manuscript values: 2.38% relative L2 error,
  22.21 s offline construction/solve, and 23.60 s prediction.
- PINN baselines in this repository use **Adam only**. All exploratory L-BFGS code was
  removed from the public release.

## 2D homogeneous canonical run

The canonical 2D homogeneous RFNN configuration uses `c=1.0`, `M=10000`,
weight half-widths `[5,4,4]`, and 10000 boundary samples per side.
The confirmed manuscript/repository relative L2 error is **0.77%**.

## Timing

Wall times are not portable. GPU warm-up, CUDA synchronization, PyTorch/BLAS versions,
and device selection can materially affect the reported values. Use the paper timings
as hardware-specific reference values rather than deterministic regression targets.

## Large data

Generated `.npy` FDTD arrays and trained `.pth` PINN weights are intentionally ignored
by Git. If desired, distribute them as GitHub Release assets rather than committing
them to the repository.
