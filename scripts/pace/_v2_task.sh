#!/bin/bash
# Single v2 array task: one seed, all requested variants, one function.
# Invoked by run_v2_chunked.sh via sbatch; all sbatch directives and the
# FUNC/DIM/VARIANTS/CLF env vars are supplied on the sbatch command line.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/pace_common.sh"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUT="${PACE_ARTIFACT_ROOT}/out/ml_v2/${FUNC}_dim${DIM}_seed${SEED}.jsonl"
mkdir -p "$(dirname "${OUT}")"

echo "v2 run: func=${FUNC} dim=${DIM} seed=${SEED} variants=${VARIANTS}"
echo "classifier=${CLF}"
echo "out=${OUT}"

"${VENV_PY}" experiment.py \
  --dim "${DIM}" \
  --seeds 1 --seed-offset "${SEED}" \
  --device cpu \
  --classifier "${CLF}" \
  --functions "${FUNC}" \
  --variants "${VARIANTS}" \
  --ml-only --ml-v2 \
  --out "${OUT}"
