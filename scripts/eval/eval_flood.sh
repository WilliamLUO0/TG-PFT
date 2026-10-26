#!/bin/bash
#SBATCH --job-name=eval_flood
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

declare -A TRAIN_DS=(
  [8]=dataset_ds8_filtered_thr0p1_min5100_full
  [2]=dataset_ds2_filtered_thr0p1_min25100_full
  [4]=dataset_ds4_filtered_thr0p1_min10100_full
  [16]=dataset_ds16_filtered_thr0p1_min2100_full
)

index_csv () {
  local scale="$1" var="$2" tag="$3"
  local suffix=""
  [[ "${scale}" != "8" ]] && suffix="_ds${scale}"
  if [[ "${tag}" == "val" ]]; then
    echo "datasets/${TRAIN_DS[$scale]}/index_with_severity_stats_${var}.csv"
  elif [[ "${tag}" == "gabrielle" ]]; then
    echo "datasets/testdataset_gabrielle${suffix}/index.csv"
  else
    echo "datasets/testdataset_${tag}0c${suffix}/index.csv"
  fi
}

eval_one () {
  local model="$1" scale="$2" var="$3" tag="$4"
  local res="results/${model}_SRx${scale}_${var}_${tag}"
  local idx
  idx="$(index_csv "${scale}" "${var}" "${tag}")"
  local abs=()
  [[ "${var}" != "h" ]] && abs=(--abs)
  python scripts/eval/eval_flood.py \
    --index-csv "${idx}" \
    --vis-root "${res}/visualization" \
    --var "${var}" "${abs[@]}" \
    --out-json "${res}/eval_summary.json" \
    --out-csv-patch "${res}/eval_patch.csv" \
    --out-csv-time "${res}/eval_time.csv" \
    --out-csv-scenario "${res}/eval_scenario.csv"
  if [[ "${model}" == "TGPFT" ]]; then
    python scripts/eval/eval_flood.py \
      --index-csv "${idx}" \
      --vis-root "${res}/visualization" \
      --var "${var}" "${abs[@]}" \
      --pred-source coarse_upsample \
      --out-json "${res}/eval_bicubic_summary.json" \
      --out-csv-time "${res}/eval_bicubic_time.csv" \
      --out-csv-scenario "${res}/eval_bicubic_scenario.csv"
  fi
}

for var in h u v; do
  for tag in val 2y42h 5y42h 10y42h 50y42h 100y42h 1000y42h gabrielle; do
    eval_one TGPFT 8 "${var}" "${tag}"
  done
done

for model in HeUNet SwinFlood RSwinUNet; do
  for tag in val 100y42h gabrielle; do
    eval_one "${model}" 8 h "${tag}"
  done
done

for scale in 2 4 16; do
  for tag in val 100y42h gabrielle; do
    eval_one TGPFT "${scale}" h "${tag}"
  done
done
