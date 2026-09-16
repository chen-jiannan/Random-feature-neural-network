#!/usr/bin/env bash
set -euo pipefail

OUT="RFNN_results_$(date +%Y%m%d_%H%M).zip"
zip -r "$OUT" \
  results_conditioning_v2 \
  results_frequency_wavenumber_v2 \
  results_long_time_v2 \
  results_2d_inclusion_v2 \
  results_2d_inclusion_lstsq_check \
  results_frequency_wr10_extension 2>/dev/null || true
echo "Created $OUT"
