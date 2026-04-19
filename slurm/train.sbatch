#!/bin/bash
#SBATCH --job-name=pso_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
mkdir -p logs

# module load anaconda/2024.02
# module load cuda/12.1

source activate pso-gpu
cd "$SLURM_SUBMIT_DIR"

python train_classifier.py \
    --data training_data.npz \
    --out stuck_classifier.lgb \
    --device cuda
