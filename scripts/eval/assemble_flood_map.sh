#!/bin/bash
#SBATCH --job-name=assemble_flood_map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
set +u
conda activate tgpft-data
set -u

export PYTHONUNBUFFERED=1

assemble () {
  local scale="$1" var="$2" tag="$3" scenario="$4"
  local suffix=""
  [[ "${scale}" != "8" ]] && suffix="_ds${scale}"
  local ds
  if [[ "${tag}" == "gabrielle" ]]; then ds="datasets/testdataset_gabrielle${suffix}"; else ds="datasets/testdataset_${tag}0c${suffix}"; fi
  local res="results/TGPFT_SRx${scale}_${var}_${tag}"
  for source in pred gt; do
    python scripts/eval/assemble_flood_map.py \
      --index-csv "${ds}/index.csv" \
      --vis-root "${res}/visualization" \
      --var "${var}" --scenario "${scenario}" \
      --source "${source}" \
      --out-dir "${res}/assembled"
  done
}

TESTSETS=(
  "2y42h:2y_42h_0c"
  "5y42h:5y_42h_0c"
  "10y42h:10y_42h_0c"
  "50y42h:50y_42h_0c"
  "100y42h:100y_42h_0c"
  "1000y42h:1000y_42h_0c"
  "gabrielle:gabrielle"
)

for var in h u v; do
  for pair in "${TESTSETS[@]}"; do
    assemble 8 "${var}" "${pair%%:*}" "${pair#*:}"
  done
done

for scale in 2 4 16; do
  assemble "${scale}" h 100y42h 100y_42h_0c
  assemble "${scale}" h gabrielle gabrielle
done
