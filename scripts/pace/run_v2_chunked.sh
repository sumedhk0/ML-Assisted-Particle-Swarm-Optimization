#!/bin/bash
# v2-only full benchmark on PACE Phoenix (embers).
#
# Pipeline: generate_data (GPU) -> train_classifier (CPU) -> v2 ML arrays (CPU).
# One job array per benchmark function, one seed per array task, so each task
# stays well under the 8h embers wall-time cap even at dim=50.
#
# Only the learned-rescue + --ml-v2 path is run (batch acquisition,
# uncertainty-aware split, adaptive ml_period). No baseline, no v1.
#
# Override any of the env vars below at the call site, e.g.:
#   DIM=20 SEEDS=21 FUNCS="sphere rastrigin" bash scripts/pace/run_v2_chunked.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "${REPO_ROOT}")"
export PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/9/${USER_NAME}/${REPO_NAME}}"
mkdir -p logs

DIM="${DIM:-50}"
SEEDS="${SEEDS:-51}"                 # seeds 0..SEEDS-1
CONCURRENCY="${CONCURRENCY:-8}"      # max simultaneous array tasks per function
VARIANTS="${VARIANTS:-A1,A2,A3,B,C1,C2}"
FUNCS="${FUNCS:-sphere rastrigin ackley griewank rosenbrock}"
ACCOUNT="${ACCOUNT:-paceship-pso}"
QOS="${QOS:-embers}"
CLF="${PACE_SCRATCH_ROOT}/stuck_classifier.lgb"

echo "Repo root:    ${REPO_ROOT}"
echo "Scratch root: ${PACE_SCRATCH_ROOT}"
echo "Classifier:   ${CLF}"
echo "Account/QOS:  ${ACCOUNT} / ${QOS}"
echo "Dim ${DIM}, seeds 0..$((SEEDS-1)), variants ${VARIANTS}"
echo "Functions:    ${FUNCS}"
echo

# --- Stage 1: generate training data (GPU) ---
DATA_JOB=$(sbatch --parsable \
  --export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},RUNS=500,ITERS=200,SAMPLE_EVERY=5,FORCE_REGEN=0 \
  slurm/generate_data.sbatch)
echo "data-gen:  ${DATA_JOB}"

# --- Stage 2: train classifier (after data succeeds) ---
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${DATA_JOB} \
  --export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},CLASSIFIER_PATH=${CLF},FORCE_RETRAIN=0 \
  slurm/train.sbatch)
echo "train:     ${TRAIN_JOB}"

# --- Stage 3: v2 ML, one array per function, one seed per task ---
V2_JOBS=()
for FUNC in ${FUNCS}; do
  JID=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
    --array=0-$((SEEDS-1))%${CONCURRENCY} \
    --job-name="v2_${FUNC}" \
    -A "${ACCOUNT}" --qos="${QOS}" -p cpu-small \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=08:00:00 \
    --output=logs/%x-%A_%a.out --error=logs/%x-%A_%a.err \
    --export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},FUNC=${FUNC},DIM=${DIM},VARIANTS=${VARIANTS},CLF=${CLF} \
    "${SCRIPT_DIR}/_v2_task.sh")
  V2_JOBS+=("${JID}")
  echo "v2 ${FUNC}: ${JID}"
done

echo
echo "All job IDs: ${DATA_JOB} ${TRAIN_JOB} ${V2_JOBS[*]}"
echo "Monitor:     squeue --me"
echo "Results ->   ${PACE_SCRATCH_ROOT}/out/ml_v2/"
