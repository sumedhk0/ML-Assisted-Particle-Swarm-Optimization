#!/bin/bash
# v2-only full benchmark on PACE Phoenix (embers).
#
# Pipeline: generate_data (GPU) -> train_classifier (CPU) -> v2 ML arrays (CPU).
# One job array per benchmark function; each array TASK runs SEEDS_PER_TASK
# seeds (all variants, one function). Seeds are packed because embers caps
# submitted jobs per user (MaxSubmitPU=50) and 1-seed-per-task would exceed it.
#
# Only the learned-rescue + --ml-v2 path is run (batch acquisition,
# uncertainty-aware split, adaptive ml_period). No baseline, no v1.
#
# Env overrides:
#   DIM, SEEDS, SEEDS_PER_TASK, CONCURRENCY, VARIANTS, FUNCS, ACCOUNT, QOS
#   TRAIN_DEP=<jobid>  Skip data-gen+train; point v2 arrays at this train job.
#
# Examples:
#   bash scripts/pace/run_v2_chunked.sh
#   TRAIN_DEP=9082582 bash scripts/pace/run_v2_chunked.sh        # reuse a train job
#   TRAIN_DEP=9082582 FUNCS=sphere SEEDS_PER_TASK=3 bash scripts/pace/run_v2_chunked.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "${REPO_ROOT}")"
export PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/9/${USER_NAME}/${REPO_NAME}}"
mkdir -p logs

DIM="${DIM:-50}"
SEEDS="${SEEDS:-51}"                       # seeds 0..SEEDS-1
SEEDS_PER_TASK="${SEEDS_PER_TASK:-6}"      # seeds packed into one array task
CONCURRENCY="${CONCURRENCY:-8}"            # max simultaneous tasks per function
VARIANTS="${VARIANTS:-A1,A2,A3,B,C1,C2}"
FUNCS="${FUNCS:-sphere rastrigin ackley griewank rosenbrock}"
ACCOUNT="${ACCOUNT:-paceship-pso}"
QOS="${QOS:-embers}"
TRAIN_DEP="${TRAIN_DEP:-}"
CLF="${PACE_SCRATCH_ROOT}/stuck_classifier.lgb"

NUM_CHUNKS=$(( (SEEDS + SEEDS_PER_TASK - 1) / SEEDS_PER_TASK ))
NUM_FUNCS=$(printf '%s\n' ${FUNCS} | wc -l)
PROJECTED=$(( NUM_FUNCS * NUM_CHUNKS ))

echo "Repo root:    ${REPO_ROOT}"
echo "Scratch root: ${PACE_SCRATCH_ROOT}"
echo "Classifier:   ${CLF}"
echo "Account/QOS:  ${ACCOUNT} / ${QOS}"
echo "Dim ${DIM}, seeds 0..$((SEEDS-1)), ${SEEDS_PER_TASK} seeds/task -> ${NUM_CHUNKS} tasks/function"
echo "Functions:    ${FUNCS}"
echo "Variants:     ${VARIANTS}"
echo "Projected v2 array tasks: ${PROJECTED} (embers MaxSubmitPU=50)"
if [[ -z "${TRAIN_DEP}" ]]; then
  echo "  + 2 data/train jobs = $((PROJECTED + 2)) total"
fi
echo

# Dependency handling:
#   TRAIN_DEP unset  -> submit data-gen + train, depend on the train job
#   TRAIN_DEP=none   -> no dependency (classifier already exists at ${CLF})
#   TRAIN_DEP=<jobid>-> depend on that existing train job
DEP_ARGS=()
if [[ -z "${TRAIN_DEP}" ]]; then
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
  DEP_ARGS=(--dependency=afterok:${TRAIN_JOB})
elif [[ "${TRAIN_DEP}" == "none" ]]; then
  echo "No train dependency; expecting classifier already present at ${CLF}"
else
  echo "Reusing existing train job ${TRAIN_DEP} (skipping data-gen + train)"
  DEP_ARGS=(--dependency=afterok:${TRAIN_DEP})
fi

# Export so sbatch --export=ALL carries them. VARIANTS contains commas, which
# are the --export list delimiter, so it MUST ride in the environment rather
# than be listed inline (otherwise sbatch reads only "A1" and drops the rest).
export PACE_SCRATCH_ROOT DIM SEEDS SEEDS_PER_TASK VARIANTS CLF

# --- Stage 3: v2 ML, one array per function, SEEDS_PER_TASK seeds per task ---
V2_JOBS=()
for FUNC in ${FUNCS}; do
  export FUNC
  JID=$(sbatch --parsable "${DEP_ARGS[@]}" \
    --array=0-$((NUM_CHUNKS-1))%${CONCURRENCY} \
    --job-name="v2_${FUNC}" \
    -A "${ACCOUNT}" --qos="${QOS}" -p cpu-small --requeue \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=08:00:00 \
    --output=logs/%x-%A_%a.out --error=logs/%x-%A_%a.err \
    --export=ALL \
    "${SCRIPT_DIR}/_v2_task.sh")
  V2_JOBS+=("${JID}")
  echo "v2 ${FUNC}: ${JID}"
done

echo
echo "v2 array job IDs: ${V2_JOBS[*]}"
echo "Monitor:          squeue --me"
echo "Results ->        ${PACE_SCRATCH_ROOT}/out/ml_v2/"
