#!/bin/bash
#SBATCH --job-name=pso_gen_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_data_%j.out
#SBATCH --error=logs/gen_data_%j.err

set -euo pipefail
mkdir -p logs

# Adjust these three lines to match your cluster's module system
# module load anaconda/2024.02
# module load cuda/12.1

source activate pso-gpu

cd "$SLURM_SUBMIT_DIR"

python generate_training_data.py \
    --runs "${RUNS:-500}" \
    --device cuda:0 \
    --out training_data.npz
