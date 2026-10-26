#!/bin/bash
#SBATCH --job-name=compute_topo_attrs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
set +u
conda activate tgpft-data
set -u

INPUT_DIR=/path/to/bgflood/input_files

python scripts/data/compute_topo_attrs.py \
  --input "${INPUT_DIR}/Elevation.nc" \
  --var Band1 \
  --outdir "${INPUT_DIR}/Topo_Attrs" \
  --dx 8
