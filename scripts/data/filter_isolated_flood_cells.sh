#!/bin/bash
#SBATCH --job-name=filter_isolated_flood_cells
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
set +u
conda activate tgpft-data
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

RESULTS_DIR=/path/to/bgflood/results
THRESHOLD=0.1
CONNECTIVITY=8
declare -A MINPATCH=( [dx8]=100 [dx16]=25 [dx32]=10 [dx64]=5 [dx128]=2 )

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
  JOBS+=("${scen}:dx8 dx16 dx32 dx64 dx128")
done
JOBS+=(
  "2y_42h_0c:dx8 dx64"
  "5y_42h_0c:dx8 dx64"
  "10y_42h_0c:dx8 dx64"
  "50y_42h_0c:dx8 dx64"
  "100y_42h_0c:dx8 dx16 dx32 dx64 dx128"
  "1000y_42h_0c:dx8 dx64"
  "gabrielle:dx8 dx16 dx32 dx64 dx128"
)

filter_one () {
  local scen="$1" res="$2"
  local minp="${MINPATCH[$res]}"
  if [[ "${res}" == "dx8" ]]; then
    local fine_dir="${RESULTS_DIR}/${scen}/dx8"
    python scripts/data/filter_isolated_flood_cells.py \
      --merge-from "${fine_dir}/BGout.nc" \
      --input-dir "${fine_dir}/per_timestep_merged" \
      --output-dir "${fine_dir}/per_timestep_merged_filtered_thr0p1_min${minp}" \
      --threshold "${THRESHOLD}" --min-patch-size "${minp}" \
      --connectivity "${CONNECTIVITY}" --overwrite
  else
    local coarse_dir="${RESULTS_DIR}/${scen}/${res}"
    python scripts/data/filter_isolated_flood_cells.py \
      --input-nc "${coarse_dir}/BGout.nc" \
      --output-nc "${coarse_dir}/BGout_filtered_thr0p1_min${minp}.nc" \
      --threshold "${THRESHOLD}" --min-patch-size "${minp}" \
      --connectivity "${CONNECTIVITY}" --overwrite
  fi
}

for job in "${JOBS[@]}"; do
  scen="${job%%:*}"
  read -r -a resolutions <<< "${job#*:}"
  for res in "${resolutions[@]}"; do
    filter_one "${scen}" "${res}"
  done
done
