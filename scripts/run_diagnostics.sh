#!/usr/bin/env bash
set -euo pipefail

python diagnostics/conditioning_solver_study.py --preset paper --device cuda
python diagnostics/frequency_wavenumber_sweep.py --preset paper --device cuda
python diagnostics/long_time_propagation.py --preset paper --device cuda --protocol fixed_budget
python diagnostics/piecewise_inclusion.py --preset paper --device cuda --mode sweep --run-partitioned
