#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/results/pace"
SUBMISSION_DIR="${RESULT_ROOT}/submissions"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SUBMISSION_DIR}/submit_smoke_sphere_${RUN_ID}.log"
MANIFEST_FILE="${SUBMISSION_DIR}/submit_smoke_sphere.tsv"
USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "${REPO_ROOT}")"
PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}/smoke/sphere_${RUN_ID}}"
PACE_VENV_DIR="${PACE_VENV_DIR:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}/.venv}"

RUNS="${PACE_RUNS:-10}"
ITERS="${PACE_ITERS:-20}"
SAMPLE_EVERY="${PACE_SAMPLE_EVERY:-5}"
DIM="${PACE_DIM:-5}"
SEEDS="${PACE_SEEDS:-1}"
BASELINE_VARIANTS="${PACE_BASELINE_VARIANTS:-${PACE_VARIANTS:-PSO,A1,B,C1}}"
ML_VARIANTS="${PACE_ML_VARIANTS:-${PACE_VARIANTS:-A1,B,C1}}"
MAX_EVALS="${PACE_MAX_EVALS:-300}"
MAX_WALL_TIME_SEC="${PACE_MAX_WALL_TIME_SEC:-}"
TRACE_DIR="${PACE_TRACE_DIR:-}"
TRACE_EVERY="${PACE_TRACE_EVERY:-}"
CURVE_DIR="${PACE_CURVE_DIR:-}"
TARGET_VALUES="${PACE_TARGET_VALUES:-}"
RESCUE_POLICIES="${PACE_RESCUE_POLICIES:-}"
EXPERIMENT_FUNCTIONS="${PACE_FUNCTIONS:-sphere}"
TRAIN_FUNCTIONS="${PACE_TRAIN_FUNCTIONS:-}"
TRAIN_DIMS="${PACE_TRAIN_DIMS:-${DIM}}"
CLASSIFIER_PATH="${PACE_CLASSIFIER_PATH:-${PACE_SCRATCH_ROOT}/stuck_classifier.lgb}"

if [[ "${SEEDS}" -lt 1 ]]; then
  echo "ERROR: PACE_SEEDS must be >= 1" >&2
  exit 1
fi

ARRAY_RANGE="0-$((SEEDS - 1))%1"

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
echo "Scratch root: ${PACE_SCRATCH_ROOT}"
echo "Shared venv: ${PACE_VENV_DIR}"
echo "Experiment functions: ${EXPERIMENT_FUNCTIONS}"
echo "Training functions: ${TRAIN_FUNCTIONS:-<default>}"
echo "Training dims: ${TRAIN_DIMS}"
echo "Baseline variants: ${BASELINE_VARIANTS}"
echo "ML variants: ${ML_VARIANTS}"
echo "Dim: ${DIM}"
echo "Runs: ${RUNS}"
echo "Iters: ${ITERS}"
echo "Sample every: ${SAMPLE_EVERY}"
echo "Max evals: ${MAX_EVALS}"
echo "Max wall time: ${MAX_WALL_TIME_SEC:-<disabled>}"
echo "Trace dir: ${TRACE_DIR:-<disabled>}"
echo "Trace every: ${TRACE_EVERY:-<default>}"
echo "Curve dir: ${CURVE_DIR:-<default per slurm job>}"
echo "Target values: ${TARGET_VALUES:-<disabled>}"
echo "Rescue policies override: ${RESCUE_POLICIES:-<default from ML_MODE>}"
echo "Seeds: ${SEEDS}"
echo "Array range: ${ARRAY_RANGE}"

export MAX_EVALS
export MAX_WALL_TIME_SEC
export DIMS="${TRAIN_DIMS}"
export FUNCTIONS="${TRAIN_FUNCTIONS}"
export TARGET_VALUES
export RESCUE_POLICIES

submit_job \
  "smoke-generate-data" \
  "slurm/generate_data.sbatch" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},RUNS=${RUNS},ITERS=${ITERS},SAMPLE_EVERY=${SAMPLE_EVERY},FORCE_REGEN=1"
DATA_JOB_ID="${LAST_JOB_ID}"

submit_job \
  "smoke-train-classifier" \
  "slurm/train.sbatch" \
  "--dependency=afterok:${DATA_JOB_ID}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},CLASSIFIER_PATH=${CLASSIFIER_PATH},FORCE_RETRAIN=1"
TRAIN_JOB_ID="${LAST_JOB_ID}"

export FUNCTIONS="${EXPERIMENT_FUNCTIONS}"

export VARIANTS="${BASELINE_VARIANTS}"
submit_job \
  "smoke-baseline" \
  "slurm/experiment.sbatch" \
  "--array=${ARRAY_RANGE}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},DIM=${DIM},ML_MODE=baseline,RESULT_TAG=smoke-baseline,FORCE_RERUN=1,TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"
BASELINE_JOB_ID="${LAST_JOB_ID}"

export VARIANTS="${ML_VARIANTS}"
submit_job \
  "smoke-ml" \
  "slurm/experiment.sbatch" \
  "--array=${ARRAY_RANGE}" \
  "--dependency=afterok:${TRAIN_JOB_ID}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},DIM=${DIM},ML_MODE=ml,CLASSIFIER_PATH=${CLASSIFIER_PATH},RESULT_TAG=smoke-ml,FORCE_RERUN=1,TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"
ML_JOB_ID="${LAST_JOB_ID}"

echo
echo "Submitted smoke jobs:"
echo "  data:      ${DATA_JOB_ID}"
echo "  train:     ${TRAIN_JOB_ID}"
echo "  baseline:  ${BASELINE_JOB_ID}"
echo "  ml:        ${ML_JOB_ID}"
echo
echo "Monitor with:"
echo "  squeue -j ${DATA_JOB_ID},${TRAIN_JOB_ID},${BASELINE_JOB_ID},${ML_JOB_ID}"
echo
echo "Artifacts:"
echo "  ${PACE_SCRATCH_ROOT}"
