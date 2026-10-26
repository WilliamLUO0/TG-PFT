#!/bin/bash
#SBATCH --job-name=make_patches
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
conda activate tgpft-data
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export GDAL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export RASTERIO_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

RESULTS_DIR=/path/to/bgflood/results
INPUT_DIR=/path/to/bgflood/input_files

SCENARIOS=(
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

FACTORS=(
  "8:dx64:5:ds8:h u v"
  "2:dx16:25:ds2:h"
  "4:dx32:10:ds4:h"
  "16:dx128:2:ds16:h"
)

for spec in "${FACTORS[@]}"; do
  IFS=":" read -r scale coarse_dx coarse_min ds_tag var_list <<< "${spec}"
  read -r -a vars <<< "${var_list}"
  python scripts/data/make_patches.py \
    --var "${vars[@]}" \
    --scenarios "${SCENARIOS[@]}" \
    --fine-template "${RESULTS_DIR}/{scenario}/dx8/per_timestep_merged_filtered_thr0p1_min100" \
    --coarse-template "${RESULTS_DIR}/{scenario}/${coarse_dx}/BGout_filtered_thr0p1_min${coarse_min}.nc" \
    --file-elev "${INPUT_DIR}/Elevation.nc" \
    --file-rough "${INPUT_DIR}/Roughness.nc" \
    --file-slope "${INPUT_DIR}/Topo_Attrs/Slope_Deg.nc" \
    --file-twi "${INPUT_DIR}/Topo_Attrs/TWI.nc" \
    --file-aspect-sin "${INPUT_DIR}/Topo_Attrs/Aspect_SIN.nc" \
    --file-aspect-cos "${INPUT_DIR}/Topo_Attrs/Aspect_COS.nc" \
    --aoi "${INPUT_DIR}/Gisborne_basin.shp" \
    --scale "${scale}" --patch-coarse 64 \
    --filter-enable --filter-thresh 0.2 \
    --out-dir "datasets/dataset_${ds_tag}_filtered_thr0p1_min${coarse_min}100_full" \
    --depth-eps 5e-5 --vel-eps 1e-5
done
