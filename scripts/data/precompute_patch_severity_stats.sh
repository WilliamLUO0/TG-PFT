#!/bin/bash
#SBATCH --job-name=precompute_patch_severity_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
set +u
conda activate tgpft
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

DATASETS=(
  "dataset_ds8_filtered_thr0p1_min5100_full:h u v"
  "dataset_ds2_filtered_thr0p1_min25100_full:h"
  "dataset_ds4_filtered_thr0p1_min10100_full:h"
  "dataset_ds16_filtered_thr0p1_min2100_full:h"
)

for spec in "${DATASETS[@]}"; do
  dsdir="datasets/${spec%%:*}"
  read -r -a vars <<< "${spec#*:}"
  for var in "${vars[@]}"; do
    if [[ "${var}" == "h" ]]; then split_json="${dsdir}/split_stats_h_asinh.json"; else split_json="${dsdir}/split_stats_${var}.json"; fi
    python scripts/data/precompute_patch_severity_stats.py \
      --index-csv "${dsdir}/index.csv" \
      --root "${dsdir}" \
      --out-csv "${dsdir}/index_with_severity_stats_${var}.csv" \
      --out-json "${dsdir}/patch_severity_summary_${var}.json" \
      --target-var "${var}" \
      --h-slight 0.1 \
      --h-severe 0.5 \
      --h-extreme 1.0 \
      --split-stats-json "${split_json}" \
      --summary-split train
  done
done
