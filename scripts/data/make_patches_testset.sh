#!/bin/bash
#SBATCH --job-name=make_patches_testset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

JOBS=(
  "2y_42h_0c:8:dx64:5::h u v"
  "5y_42h_0c:8:dx64:5::h u v"
  "10y_42h_0c:8:dx64:5::h u v"
  "50y_42h_0c:8:dx64:5::h u v"
  "100y_42h_0c:8:dx64:5::h u v"
  "1000y_42h_0c:8:dx64:5::h u v"
  "gabrielle:8:dx64:5::h u v"
  "100y_42h_0c:2:dx16:25:_ds2:h"
  "100y_42h_0c:4:dx32:10:_ds4:h"
  "100y_42h_0c:16:dx128:2:_ds16:h"
  "gabrielle:2:dx16:25:_ds2:h"
  "gabrielle:4:dx32:10:_ds4:h"
  "gabrielle:16:dx128:2:_ds16:h"
)

for spec in "${JOBS[@]}"; do
  IFS=":" read -r scen scale coarse_dx coarse_min suffix var_list <<< "${spec}"
  read -r -a vars <<< "${var_list}"
  python scripts/data/make_patches.py \
    --var "${vars[@]}" \
    --scenarios "${scen}" \
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
    --filter-enable --filter-thresh 0.000001 \
    --out-dir "datasets/testdataset_${scen//_/}${suffix}" \
    --depth-eps 5e-5 --vel-eps 1e-5
done
