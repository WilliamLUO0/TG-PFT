#!/bin/bash
#SBATCH --job-name=train_TGPFT_SRx8_h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
set +u
conda activate tgpft
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nvidia-smi

srun torchrun --nproc_per_node=4 --standalone \
  basicsr/train_flood_map.py -opt options/train/train_TGPFT_SRx8_h.yml \
  --launcher pytorch --auto_resume
