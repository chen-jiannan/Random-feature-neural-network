# Generated reference data

The paper-case RFNN and PINN scripts expect the following files in this directory:

- `1d_fdtd_two_sources_cos_1e-4.npy`
- `2d_fdtd_two_source_2e-3.npy`
- `3d_fdtd_three_sources_2.5e-3.npy`
- `2d_fdtd_ref_2e-4.npy` (2D damping reference)

The first three can be generated with the homogeneous FDTD scripts in
`paper_cases/baselines/`.

The 2D damping reference is intentionally not bundled because it is large and the
original workflow used a separately generated enlarged-domain reference. If the
repository is released with precomputed data, place that array in GitHub Release assets
and document the download URL here.

`.npy` files are ignored by the repository `.gitignore`.
