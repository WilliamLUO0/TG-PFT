#!/bin/bash
set -euo pipefail

TRAIN_SCENARIOS=(
  2y_6h_0c    2y_48h_0c
  5y_6h_0c    5y_48h_0c
  10y_6h_0c   10y_48h_0c
  20y_6h_0c   20y_48h_0c
  50y_6h_0c   50y_48h_0c
  100y_6h_0c  100y_48h_0c
  200y_6h_0c  200y_48h_0c
  500y_6h_0c  500y_48h_0c
  1000y_6h_0c 1000y_48h_0c
)

JOBS=()
for scen in "${TRAIN_SCENARIOS[@]}"; do
  JOBS+=("${scen}:8 16 32 64 128")
done
JOBS+=(
  "2y_42h_0c:8 64"
  "5y_42h_0c:8 64"
  "10y_42h_0c:8 64"
  "50y_42h_0c:8 64"
  "100y_42h_0c:8 16 32 64 128"
  "1000y_42h_0c:8 64"
  "gabrielle:8 16 32 64 128"
)

for job in "${JOBS[@]}"; do
  scen="${job%%:*}"
  read -r -a dxs <<< "${job#*:}"
  for dx in "${dxs[@]}"; do
    sbatch --job-name="bgflood_${scen}_dx${dx}" scripts/bgflood/run_bgflood.sh "${scen}" "${dx}"
  done
done
