#!/bin/bash
#SBATCH --job-name=precompute_split_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
  dataset_ds8_filtered_thr0p1_min5100_full
  dataset_ds2_filtered_thr0p1_min25100_full
  dataset_ds4_filtered_thr0p1_min10100_full
  dataset_ds16_filtered_thr0p1_min2100_full
)

for ds in "${DATASETS[@]}"; do
  dsdir="datasets/${ds}"
  python scripts/data/precompute_split_stats.py \
    --index_csv "${dsdir}/index.csv" \
    --root "${dsdir}" \
    --out_json "${dsdir}/split_stats_h_asinh.json" \
    --target_var h \
    --aux_vars zs \
    --h_tau 0.1 \
    --h_q_list 50,75,90,95,99 \
    --compute_h_flood_intervals \
    --h_flood_interval_thresholds 0.1,0.5,1.0 \
    --by scenario --val_ratio 0.2 --seed 61 \
    --bins 8192

  if [[ "${ds}" == *ds8_* ]]; then
    for var in u v; do
      python scripts/data/precompute_split_stats.py \
        --index_csv "${dsdir}/index.csv" \
        --root "${dsdir}" \
        --out_json "${dsdir}/split_stats_${var}.json" \
        --target_var "${var}" \
        --reuse_split_from "${dsdir}/split_stats_h_asinh.json" \
        --uv_tau 0.1 \
        --compute_uv_intervals \
        --uv_interval_thresholds 0.1,0.5,1.0 \
        --by scenario --val_ratio 0.2 --seed 61 \
        --bins 8192
    done
  fi
done
