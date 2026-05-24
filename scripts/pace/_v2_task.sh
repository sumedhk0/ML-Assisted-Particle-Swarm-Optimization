#!/bin/bash
# Single v2 array task: SEEDS_PER_TASK seeds, all requested variants, one
# function. Invoked by run_v2_chunked.sh via sbatch; all sbatch directives and
# the FUNC/DIM/SEEDS/SEEDS_PER_TASK/VARIANTS/CLF env vars are supplied on the
# sbatch command line. The array task index selects the seed chunk.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/pace_common.sh"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SPT="${SEEDS_PER_TASK:-1}"
SEED_BASE=$(( TASK * SPT ))
REMAINING=$(( SEEDS - SEED_BASE ))
if (( REMAINING <= 0 )); then
  echo "task ${TASK}: no seeds in range (SEED_BASE=${SEED_BASE} >= SEEDS=${SEEDS}); nothing to do"
  exit 0
fi
N=$SPT
if (( N > REMAINING )); then
  N=$REMAINING
fi
SEED_END=$(( SEED_BASE + N - 1 ))

OUT="${PACE_ARTIFACT_ROOT}/out/ml_v2/${FUNC}_dim${DIM}_seeds${SEED_BASE}-${SEED_END}.jsonl"
mkdir -p "$(dirname "${OUT}")"

echo "v2 run: func=${FUNC} dim=${DIM} seeds=${SEED_BASE}..${SEED_END} (${N}) variants=${VARIANTS}"
echo "classifier=${CLF}"
echo "out=${OUT}"

"${VENV_PY}" experiment.py \
  --dim "${DIM}" \
  --seeds "${N}" --seed-offset "${SEED_BASE}" \
  --device cpu \
  --classifier "${CLF}" \
  --functions "${FUNC}" \
  --variants "${VARIANTS}" \
  --ml-only --ml-v2 \
  --out "${OUT}"
