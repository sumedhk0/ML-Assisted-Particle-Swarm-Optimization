#!/usr/bin/env bash
# End-to-end pipeline for a single machine (Linux or WSL2 on Windows).
# Usage: bash scripts/run_local.sh [DIM] [SEEDS] [DEVICE]
set -euo pipefail

DIM=${1:-10}
SEEDS=${2:-51}
DEVICE=${3:-cuda:0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">>> CUDA visibility check"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

if [[ ! -f training_data.npz ]]; then
    echo ">>> Generating training data (one-time)"
    python generate_training_data.py --runs 500 --device "$DEVICE" --out training_data.npz
fi

if [[ ! -f stuck_classifier.lgb ]]; then
    echo ">>> Training classifier (one-time)"
    python train_classifier.py --data training_data.npz --out stuck_classifier.lgb --device cuda
fi

echo ">>> Benchmark: dim=$DIM seeds=$SEEDS"
mkdir -p out
python experiment.py --dim "$DIM" --seeds "$SEEDS" --device "$DEVICE" \
    --classifier stuck_classifier.lgb --out "out/results_dim${DIM}.jsonl"

echo ">>> Aggregated summary"
python aggregate_results.py "out/results_dim${DIM}.jsonl"
