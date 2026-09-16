# Paper-to-code map

| Manuscript component | Code |
|---|---|
| 1D homogeneous RFNN | `paper_cases/rfnn/rfnn_1d_homogeneous.py` |
| 2D homogeneous RFNN | `paper_cases/rfnn/rfnn_2d_homogeneous.py` |
| 3D homogeneous RFNN | `paper_cases/rfnn/rfnn_3d_homogeneous.py` |
| 1D damping layer | `paper_cases/rfnn/rfnn_1d_damping.py` |
| Damping seed sensitivity | `paper_cases/rfnn/rfnn_1d_seed_sensitivity.py` |
| Damping-parameter study | `paper_cases/rfnn/rfnn_1d_damping_parameter_study.py` |
| Smooth heterogeneous medium / beta sweep | `paper_cases/rfnn/rfnn_1d_heterogeneous.py` |
| Feature/collocation study | `paper_cases/rfnn/rfnn_1d_feature_collocation_study.py` |
| 2D damping RFNN | `paper_cases/rfnn/rfnn_2d_damping.py` |
| 1D/2D/3D FDTD references | `paper_cases/baselines/fdtd_*_homogeneous.py` |
| Adam-only PINN baselines | `paper_cases/baselines/pinn_*_adam.py` |
| Frequency/wavenumber diagnostic | `diagnostics/frequency_wavenumber_sweep.py` |
| Long-time diagnostic | `diagnostics/long_time_propagation.py` |
| Conditioning/linear-solver diagnostic | `diagnostics/conditioning_solver_study.py` |
| Piecewise inclusion diagnostic | `diagnostics/piecewise_inclusion.py` |

The large 2D damping reference file is not stored in the repository. The supplied
comparison workflow is retained in `paper_cases/baselines/fdtd_2d_damping_comparison.py`.
