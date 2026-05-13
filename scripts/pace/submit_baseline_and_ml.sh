#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/results/pace"
SUBMISSION_DIR="${RESULT_ROOT}/submissions"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SUBMISSION_DIR}/submit_baseline_and_ml_${RUN_ID}.log"
MANIFEST_FILE="${SUBMISSION_DIR}/submit_baseline_and_ml.tsv"
USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "${REPO_ROOT}")"
PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}}"

RUNS="${PACE_RUNS:-500}"
ITERS="${PACE_ITERS:-200}"
SAMPLE_EVERY="${PACE_SAMPLE_EVERY:-5}"
DIM="${PACE_DIM:-50}"
SEEDS="${PACE_SEEDS:-51}"
ARRAY_MAX_CONCURRENT="${PACE_ARRAY_MAX_CONCURRENT:-8}"
EXPERIMENT_FUNCTIONS="${PACE_FUNCTIONS:-}"
TRAIN_FUNCTIONS="${PACE_TRAIN_FUNCTIONS-${EXPERIMENT_FUNCTIONS}}"
TRAIN_DIMS="${PACE_TRAIN_DIMS:-}"
BASELINE_VARIANTS="${PACE_BASELINE_VARIANTS:-${PACE_VARIANTS:-PSO,A1,A2,A3,B,C1,C2}}"
ML_VARIANTS="${PACE_ML_VARIANTS:-${PACE_VARIANTS:-A1,A2,A3,B,C1,C2}}"
MAX_EVALS="${PACE_MAX_EVALS:-}"
MAX_WALL_TIME_SEC="${PACE_MAX_WALL_TIME_SEC:-}"
TRACE_DIR="${PACE_TRACE_DIR:-}"
TRACE_EVERY="${PACE_TRACE_EVERY:-}"
CURVE_DIR="${PACE_CURVE_DIR:-}"
TARGET_VALUES="${PACE_TARGET_VALUES:-}"
RESCUE_POLICIES="${PACE_RESCUE_POLICIES:-}"
CLASSIFIER_PATH="${PACE_CLASSIFIER_PATH:-${PACE_SCRATCH_ROOT}/stuck_classifier.lgb}"
FORCE_REGEN="${PACE_FORCE_REGEN:-0}"
FORCE_RETRAIN="${PACE_FORCE_RETRAIN:-0}"
FORCE_RERUN="${PACE_FORCE_RERUN:-0}"

if [[ "${SEEDS}" -lt 1 ]]; then
  echo "ERROR: PACE_SEEDS must be >= 1" >&2
  exit 1
fi

if [[ "${ARRAY_MAX_CONCURRENT}" -lt 1 ]]; then
  echo "ERROR: PACE_ARRAY_MAX_CONCURRENT must be >= 1" >&2
  exit 1
fi

ARRAY_RANGE="0-$((SEEDS - 1))%${ARRAY_MAX_CONCURRENT}"

mkdir -p "${SUBMISSION_DIR}" "${REPO_ROOT}/logs"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

if [[ ! -f "${MANIFEST_FILE}" ]]; then
  printf 'timestamp\tlabel\tsbatch_command\tjob_id\n' > "${MANIFEST_FILE}"
fi

LAST_JOB_ID=""

submit_job() {
  local label="$1"
  local script_path="$2"
  shift 2

  local -a cmd=(sbatch --parsable)
  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi
  cmd+=("${script_path}")

  local cmd_string
  local job_id
  local exit_code

  cmd_string="$(printf '%q ' "${cmd[@]}")"
  cmd_string="${cmd_string% }"

  echo "[$(date '+%F %T')] SUBMIT: ${label}"
  echo "CMD: ${cmd_string}"

  if job_id="$("${cmd[@]}")"; then
    LAST_JOB_ID="${job_id}"
    echo "JOB_ID: ${job_id}"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${label}" "${cmd_string}" "${job_id}" >> "${MANIFEST_FILE}"
  else
    exit_code=$?
    echo "JOB_SUBMISSION_FAILED: exit_code=${exit_code}"
    printf '%s\t%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${label}" "${cmd_string}" "${exit_code}" >> "${MANIFEST_FILE}"
    return "${exit_code}"
  fi
}

echo "Repository root: ${REPO_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Manifest: ${MANIFEST_FILE}"
echo "Runs: ${RUNS}"
echo "Iters: ${ITERS}"
echo "Sample every: ${SAMPLE_EVERY}"
echo "Dim: ${DIM}"
echo "Seeds: ${SEEDS}"
echo "Array range: ${ARRAY_RANGE}"
echo "Scratch root: ${PACE_SCRATCH_ROOT}"
echo "Classifier path: ${CLASSIFIER_PATH}"
echo "Experiment functions: ${EXPERIMENT_FUNCTIONS:-<default>}"
echo "Training functions: ${TRAIN_FUNCTIONS:-<default>}"
echo "Training dims: ${TRAIN_DIMS:-<default>}"
echo "Baseline variants: ${BASELINE_VARIANTS}"
echo "ML variants: ${ML_VARIANTS}"
echo "Max evals: ${MAX_EVALS:-<default>}"
echo "Max wall time: ${MAX_WALL_TIME_SEC:-<disabled>}"
echo "Trace dir: ${TRACE_DIR:-<disabled>}"
echo "Trace every: ${TRACE_EVERY:-<default>}"
echo "Curve dir: ${CURVE_DIR:-<default per slurm job>}"
echo "Target values: ${TARGET_VALUES:-<disabled>}"
echo "Rescue policies override: ${RESCUE_POLICIES:-<default from ML_MODE>}"

export MAX_EVALS
export MAX_WALL_TIME_SEC
export DIMS="${TRAIN_DIMS}"
export FUNCTIONS="${TRAIN_FUNCTIONS}"
export TARGET_VALUES
export RESCUE_POLICIES

submit_job \
  "generate-data" \
  "slurm/generate_data.sbatch" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},RUNS=${RUNS},ITERS=${ITERS},SAMPLE_EVERY=${SAMPLE_EVERY},FORCE_REGEN=${FORCE_REGEN}"
DATA_JOB_ID="${LAST_JOB_ID}"

submit_job \
  "train-classifier" \
  "slurm/train.sbatch" \
  "--dependency=afterok:${DATA_JOB_ID}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},CLASSIFIER_PATH=${CLASSIFIER_PATH},FORCE_RETRAIN=${FORCE_RETRAIN}"
TRAIN_JOB_ID="${LAST_JOB_ID}"

export FUNCTIONS="${EXPERIMENT_FUNCTIONS}"

export VARIANTS="${BASELINE_VARIANTS}"
submit_job \
  "baseline-array" \
  "slurm/experiment.sbatch" \
  "--array=${ARRAY_RANGE}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},DIM=${DIM},ML_MODE=baseline,RESULT_TAG=baseline,FORCE_RERUN=${FORCE_RERUN},TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"
BASELINE_JOB_ID="${LAST_JOB_ID}"

export VARIANTS="${ML_VARIANTS}"
submit_job \
  "ml-array" \
  "slurm/experiment.sbatch" \
  "--array=${ARRAY_RANGE}" \
  "--dependency=afterok:${TRAIN_JOB_ID}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},DIM=${DIM},ML_MODE=ml,CLASSIFIER_PATH=${CLASSIFIER_PATH},RESULT_TAG=ml,FORCE_RERUN=${FORCE_RERUN},TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"
ML_JOB_ID="${LAST_JOB_ID}"

echo
echo "Submitted jobs:"
echo "  data:      ${DATA_JOB_ID}"
echo "  train:     ${TRAIN_JOB_ID}"
echo "  baseline:  ${BASELINE_JOB_ID}"
echo "  ml:        ${ML_JOB_ID}"
echo
echo "Monitor with:"
echo "  squeue -j ${DATA_JOB_ID},${TRAIN_JOB_ID},${BASELINE_JOB_ID},${ML_JOB_ID}"
