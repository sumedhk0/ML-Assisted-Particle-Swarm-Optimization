#!/usr/bin/env bash

if [[ -n "${_PSO_PACE_COMMON_SH:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
_PSO_PACE_COMMON_SH=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"
USER_NAME="${USER:-$(id -un)}"
PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}}"
PACE_ARTIFACT_ROOT="${PACE_ARTIFACT_ROOT:-${PACE_SCRATCH_ROOT}}"
PACE_VENV_DIR="${PACE_VENV_DIR:-${PACE_SCRATCH_ROOT}/.venv}"
PACE_TMPDIR="${PACE_TMPDIR:-${PACE_SCRATCH_ROOT}/tmp}"
VENV_PY="${PACE_VENV_DIR}/bin/python"

cd "${REPO_ROOT}"
mkdir -p logs out results/pace
mkdir -p "${PACE_ARTIFACT_ROOT}" "${PACE_TMPDIR}"

unset MLFLOW_TRACKING_URI
export PYTHONUNBUFFERED=1
export TMPDIR="${PACE_TMPDIR}"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

if [[ "${PACE_SKIP_CUDA_MODULE:-0}" != "1" ]] && declare -F module >/dev/null 2>&1; then
  module load cuda/12.6.1 >/dev/null 2>&1 || true
fi

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: expected virtualenv python not found at ${VENV_PY}" >&2
  echo "Run: bash slurm/install_env.sh --python python3 --venv ${PACE_VENV_DIR}" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${PACE_VENV_DIR}/bin/activate"
