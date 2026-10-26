#!/bin/bash
#SBATCH --job-name=bgflood
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCENARIO=${1:?usage: run_bgflood.sh <scenario> <dx>}
DX=${2:?usage: run_bgflood.sh <scenario> <dx>}

BG_FLOOD=/path/to/BG_Flood/BG_Flood
INPUT_DIR=/path/to/bgflood/input_files
RESULTS_DIR=/path/to/bgflood/results
PARAM_DIR=${PARAM_DIR:-scripts/bgflood}

case "${DX}" in
  8)               BASE_DX=128;     MAXLEVEL=4 ;;
  16|32|64|128)    BASE_DX="${DX}"; MAXLEVEL=0 ;;
  *) echo "[ERROR] unsupported dx: ${DX}" >&2; exit 1 ;;
esac

if [[ "${SCENARIO}" == "gabrielle" ]]; then
  TEMPLATE="${PARAM_DIR}/BG_param_gabrielle.txt"
  RAINFILE=""
else
  TEMPLATE="${PARAM_DIR}/BG_param_design.txt"
  RAINFILE="${INPUT_DIR}/${SCENARIO}/${SCENARIO}_nztm.nc?depth"
fi
TEMPLATE="$(realpath "${TEMPLATE}")"

RESULT_DIR="${RESULTS_DIR}/${SCENARIO}/dx${DX}"
if [[ -s "${RESULT_DIR}/BGout.nc" ]]; then
  echo "[skip] ${RESULT_DIR}/BGout.nc already exists"
  exit 0
fi
mkdir -p "${RESULT_DIR}"
cd "${RESULT_DIR}"

awk -v input_dir="${INPUT_DIR}" -v rainfile="${RAINFILE}" -v dx="${BASE_DX}" -v maxlevel="${MAXLEVEL}" '
  { gsub("@INPUT_DIR@", input_dir) }
  /^[[:space:]]*rainfile[[:space:]]*=/ && rainfile != "" { print "rainfile = " rainfile; next }
  /^[[:space:]]*dx[[:space:]]*=/ { printf "dx = %.6f\n", dx; next }
  /^[[:space:]]*maxlevel[[:space:]]*=/ { printf "maxlevel = %d\n", maxlevel; next }
  { print }
' "${TEMPLATE}" > BG_param.txt

echo "[info] scenario=${SCENARIO} dx=${DX} (base dx=${BASE_DX}, maxlevel=${MAXLEVEL}) -> ${RESULT_DIR}"

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
srun "${BG_FLOOD}" BG_param.txt
