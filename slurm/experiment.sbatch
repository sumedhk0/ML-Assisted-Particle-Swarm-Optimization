#!/bin/bash
#SBATCH --job-name=pso_exp
#SBATCH --array=0-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# Array job: one GPU task per seed. DIM defaults to 50; override with
#   sbatch --export=DIM=100,ALL slurm/experiment.sbatch
set -euo pipefail
mkdir -p logs out

# module load anaconda/2024.02
# module load cuda/12.1

source activate pso-gpu
cd "$SLURM_SUBMIT_DIR"

DIM="${DIM:-50}"
SEED="$SLURM_ARRAY_TASK_ID"

python experiment.py \
    --dim "$DIM" \
    --seeds 1 \
    --seed-offset "$SEED" \
    --device cuda:0 \
    --classifier stuck_classifier.lgb \
    --out "out/results_dim${DIM}_seed${SEED}.jsonl"
