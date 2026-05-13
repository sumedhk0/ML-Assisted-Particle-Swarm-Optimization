#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/results/pace"
SUBMISSION_DIR="${RESULT_ROOT}/submissions"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SUBMISSION_DIR}/submit_single_problem_chunked_${RUN_ID}.log"
MANIFEST_FILE="${SUBMISSION_DIR}/submit_single_problem_chunked.tsv"
USER_NAME="${USER:-$(id -un)}"
REPO_NAME="$(basename "${REPO_ROOT}")"

EXPERIMENT_FUNCTIONS="${PACE_FUNCTIONS:-rastrigin}"
SAFE_LABEL="${EXPERIMENT_FUNCTIONS//,/__}"
PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}/runs/${SAFE_LABEL}_${RUN_ID}}"
PACE_VENV_DIR="${PACE_VENV_DIR:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}/.venv}"
TRAIN_FUNCTIONS="${PACE_TRAIN_FUNCTIONS:-}"

RUNS="${PACE_RUNS:-500}"
ITERS="${PACE_ITERS:-200}"
SAMPLE_EVERY="${PACE_SAMPLE_EVERY:-5}"
DIM="${PACE_DIM:-50}"
SEEDS="${PACE_SEEDS:-51}"
SEED_CHUNK_SIZE="${PACE_SEED_CHUNK_SIZE:-8}"
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

default_target_values_for_problem() {
  case "$1" in
    sphere)
      printf '1.0,0.5'
      ;;
    ackley)
      printf '1.0,0.5'
      ;;
    rastrigin)
      printf '100,75'
      ;;
    griewank)
      printf '100,50'
      ;;
    rosenbrock)
      printf '50000,10000'
      ;;
    *)
      printf ''
      ;;
  esac
}

if [[ -z "${EXPERIMENT_FUNCTIONS}" ]]; then
  echo "ERROR: PACE_FUNCTIONS must name at least one benchmark function" >&2
  exit 1
fi

if [[ -z "${TARGET_VALUES}" && "${EXPERIMENT_FUNCTIONS}" != *","* ]]; then
  TARGET_VALUES="$(default_target_values_for_problem "${EXPERIMENT_FUNCTIONS}")"
fi

if [[ "${SEEDS}" -lt 1 ]]; then
  echo "ERROR: PACE_SEEDS must be >= 1" >&2
  exit 1
fi

if [[ "${SEED_CHUNK_SIZE}" -lt 1 ]]; then
  echo "ERROR: PACE_SEED_CHUNK_SIZE must be >= 1" >&2
  exit 1
fi

mkdir -p "${SUBMISSION_DIR}" "${REPO_ROOT}/logs"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

if [[ ! -f "${MANIFEST_FILE}" ]]; then
  printf 'timestamp\tlabel\tsbatch_command\tjob_id\n' > "${MANIFEST_FILE}"
fi

LAST_JOB_ID=""
JOB_IDS=()

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
    JOB_IDS+=("${job_id}")
    echo "JOB_ID: ${job_id}"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${label}" "${cmd_string}" "${job_id}" >> "${MANIFEST_FILE}"
  else
    exit_code=$?
    echo "JOB_SUBMISSION_FAILED: exit_code=${exit_code}"
    printf '%s\t%s\t%s\tFAILED:%s\n' "$(date '+%F %T')" "${label}" "${cmd_string}" "${exit_code}" >> "${MANIFEST_FILE}"
    return "${exit_code}"
  fi
}

join_by_comma() {
  local first=1
  local value
  for value in "$@"; do
    if [[ "${first}" -eq 1 ]]; then
      printf '%s' "${value}"
      first=0
    else
      printf ',%s' "${value}"
    fi
  done
}

echo "Repository root: ${REPO_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Manifest: ${MANIFEST_FILE}"
echo "Scratch root: ${PACE_SCRATCH_ROOT}"
echo "Shared venv: ${PACE_VENV_DIR}"
echo "Experiment functions: ${EXPERIMENT_FUNCTIONS}"
echo "Training functions: ${TRAIN_FUNCTIONS:-<default>}"
echo "Runs: ${RUNS}"
echo "Iters: ${ITERS}"
echo "Sample every: ${SAMPLE_EVERY}"
echo "Dim: ${DIM}"
echo "Seeds: ${SEEDS}"
echo "Seed chunk size: ${SEED_CHUNK_SIZE}"
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
echo "Classifier path: ${CLASSIFIER_PATH}"

export MAX_EVALS
export MAX_WALL_TIME_SEC
export DIMS="${TRAIN_DIMS}"
export FUNCTIONS="${TRAIN_FUNCTIONS}"
export TARGET_VALUES
export RESCUE_POLICIES

submit_job \
  "generate-data:${EXPERIMENT_FUNCTIONS}" \
  "slurm/generate_data.sbatch" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},RUNS=${RUNS},ITERS=${ITERS},SAMPLE_EVERY=${SAMPLE_EVERY},FORCE_REGEN=${FORCE_REGEN}"
DATA_JOB_ID="${LAST_JOB_ID}"

submit_job \
  "train-classifier:${EXPERIMENT_FUNCTIONS}" \
  "slurm/train.sbatch" \
  "--dependency=afterok:${DATA_JOB_ID}" \
  "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},CLASSIFIER_PATH=${CLASSIFIER_PATH},FORCE_RETRAIN=${FORCE_RETRAIN}"
TRAIN_JOB_ID="${LAST_JOB_ID}"

export FUNCTIONS="${EXPERIMENT_FUNCTIONS}"

for ((seed_base = 0; seed_base < SEEDS; seed_base += SEED_CHUNK_SIZE)); do
  seeds_per_job="${SEED_CHUNK_SIZE}"
  if (( seed_base + seeds_per_job > SEEDS )); then
    seeds_per_job="$((SEEDS - seed_base))"
  fi
  seed_end="$((seed_base + seeds_per_job - 1))"
  out_stem="results_dim${DIM}_seeds${seed_base}-${seed_end}"

  export VARIANTS="${BASELINE_VARIANTS}"
  submit_job \
    "baseline:${EXPERIMENT_FUNCTIONS}:${seed_base}-${seed_end}" \
    "slurm/experiment.sbatch" \
    "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},DIM=${DIM},ML_MODE=baseline,RESULT_TAG=baseline,FORCE_RERUN=${FORCE_RERUN},SEED_BASE=${seed_base},SEEDS_PER_JOB=${seeds_per_job},OUT_STEM=${out_stem},TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"

  export VARIANTS="${ML_VARIANTS}"
  submit_job \
    "ml:${EXPERIMENT_FUNCTIONS}:${seed_base}-${seed_end}" \
    "slurm/experiment.sbatch" \
    "--dependency=afterok:${TRAIN_JOB_ID}" \
    "--export=ALL,PACE_SCRATCH_ROOT=${PACE_SCRATCH_ROOT},PACE_VENV_DIR=${PACE_VENV_DIR},DIM=${DIM},ML_MODE=ml,CLASSIFIER_PATH=${CLASSIFIER_PATH},RESULT_TAG=ml,FORCE_RERUN=${FORCE_RERUN},SEED_BASE=${seed_base},SEEDS_PER_JOB=${seeds_per_job},OUT_STEM=${out_stem},TRACE_DIR=${TRACE_DIR},TRACE_EVERY=${TRACE_EVERY},CURVE_DIR=${CURVE_DIR}"
done

echo
echo "Submitted jobs:"
echo "  data:      ${DATA_JOB_ID}"
echo "  train:     ${TRAIN_JOB_ID}"
echo "  all jobs:  $(join_by_comma "${JOB_IDS[@]}")"
echo
echo "Monitor with:"
echo "  squeue -j $(join_by_comma "${JOB_IDS[@]}")"
echo
echo "Artifacts:"
echo "  ${PACE_SCRATCH_ROOT}"
