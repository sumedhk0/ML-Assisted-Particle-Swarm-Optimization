#!/bin/bash
#SBATCH --job-name=pso_experiment
#SBATCH -A gts-sn73
#SBATCH --qos=embers
#SBATCH -p cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/pace_common.sh"

DIM="${DIM:-50}"
SEED_BASE="${SEED_BASE:-${SLURM_ARRAY_TASK_ID:-0}}"
SEEDS_PER_JOB="${SEEDS_PER_JOB:-1}"
DEVICE="${PSO_DEVICE:-cpu}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-${PACE_ARTIFACT_ROOT}/stuck_classifier.lgb}"
ML_MODE="${ML_MODE:-both}"
RESULT_TAG="${RESULT_TAG:-${ML_MODE}}"
RESULT_DIR="${RESULT_DIR:-${PACE_ARTIFACT_ROOT}/out/${RESULT_TAG}}"
SEED_END="$((SEED_BASE + SEEDS_PER_JOB - 1))"
if [[ -n "${OUT_STEM:-}" ]]; then
  DEFAULT_OUT_STEM="${OUT_STEM}"
elif [[ "${SEEDS_PER_JOB}" -eq 1 ]]; then
  DEFAULT_OUT_STEM="results_dim${DIM}_seed${SEED_BASE}"
else
  DEFAULT_OUT_STEM="results_dim${DIM}_seeds${SEED_BASE}-${SEED_END}"
fi
OUT_PATH="${OUT_PATH:-${RESULT_DIR}/${DEFAULT_OUT_STEM}.jsonl}"
MAX_EVALS="${MAX_EVALS:-}"
MAX_WALL_TIME_SEC="${MAX_WALL_TIME_SEC:-}"
TRACE_DIR="${TRACE_DIR:-}"
TRACE_EVERY="${TRACE_EVERY:-}"
CURVE_DIR="${CURVE_DIR:-${PACE_ARTIFACT_ROOT}/curves/${RESULT_TAG}}"
TARGET_VALUES="${TARGET_VALUES:-}"
RESCUE_POLICIES="${RESCUE_POLICIES:-}"

mkdir -p "${RESULT_DIR}"

if [[ -f "${OUT_PATH}" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[$(date '+%F %T')] SKIP: ${OUT_PATH} already exists"
  exit 0
fi

cmd=(
  "${VENV_PY}" experiment.py
  --dim "${DIM}"
  --seeds "${SEEDS_PER_JOB}"
  --seed-offset "${SEED_BASE}"
  --device "${DEVICE}"
  --classifier "${CLASSIFIER_PATH}"
  --out "${OUT_PATH}"
)

if [[ -n "${MAX_EVALS}" ]]; then
  cmd+=(--max-evals "${MAX_EVALS}")
fi

if [[ -n "${MAX_WALL_TIME_SEC}" ]]; then
  cmd+=(--max-wall-time-sec "${MAX_WALL_TIME_SEC}")
fi

if [[ -n "${TRACE_DIR}" ]]; then
  mkdir -p "${TRACE_DIR}"
  cmd+=(--trace-dir "${TRACE_DIR}")
fi

if [[ -n "${TRACE_EVERY}" ]]; then
  cmd+=(--trace-every "${TRACE_EVERY}")
fi

if [[ -n "${CURVE_DIR}" ]]; then
  mkdir -p "${CURVE_DIR}"
  cmd+=(--curve-dir "${CURVE_DIR}")
fi

if [[ -n "${TARGET_VALUES}" ]]; then
  cmd+=(--targets "${TARGET_VALUES}")
fi

if [[ -n "${RESCUE_POLICIES}" ]]; then
  cmd+=(--rescue-policies "${RESCUE_POLICIES}")
fi

case "${ML_MODE}" in
  both)
    ;;
  baseline)
    cmd+=(--baseline-only)
    ;;
  ml)
    if [[ ! -f "${CLASSIFIER_PATH}" ]]; then
      echo "ERROR: classifier file not found at ${CLASSIFIER_PATH}" >&2
      exit 1
    fi
    cmd+=(--ml-only)
    ;;
  *)
    echo "ERROR: unsupported ML_MODE='${ML_MODE}'" >&2
    echo "Use one of: both, baseline, ml" >&2
    exit 1
    ;;
esac

if [[ -n "${FUNCTIONS:-}" ]]; then
  cmd+=(--functions "${FUNCTIONS}")
fi

if [[ -n "${VARIANTS:-}" ]]; then
  cmd+=(--variants "${VARIANTS}")
fi

echo "Dim: ${DIM}"
echo "Seed base: ${SEED_BASE}"
echo "Seeds/job: ${SEEDS_PER_JOB}"
echo "Seed end: ${SEED_END}"
echo "Device: ${DEVICE}"
echo "Classifier: ${CLASSIFIER_PATH}"
echo "ML mode: ${ML_MODE}"
echo "Output: ${OUT_PATH}"
echo "Max evals: ${MAX_EVALS:-<default>}"
echo "Max wall time: ${MAX_WALL_TIME_SEC:-<disabled>}"
echo "Trace dir: ${TRACE_DIR:-<disabled>}"
echo "Trace every: ${TRACE_EVERY:-<default>}"
echo "Curve dir: ${CURVE_DIR:-<disabled>}"
echo "Target values: ${TARGET_VALUES:-<disabled>}"
echo "Rescue policies: ${RESCUE_POLICIES:-<default>}"

"${cmd[@]}"
