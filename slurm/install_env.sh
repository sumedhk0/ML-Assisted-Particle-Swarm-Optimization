#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="$(basename "${REPO_ROOT}")"
USER_NAME="${USER:-$(id -un)}"
PACE_SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/storage/scratch1/4/${USER_NAME}/${REPO_NAME}}"
DEFAULT_PACE_VENV_DIR="${PACE_SCRATCH_ROOT}/.venv"
DEFAULT_PIP_CACHE_DIR="/storage/scratch1/4/${USER_NAME}/.cache/pip"
DEFAULT_TMPDIR="${PACE_SCRATCH_ROOT}/tmp"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PACE_VENV_DIR:-${DEFAULT_PACE_VENV_DIR}}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DEFAULT_PIP_CACHE_DIR}}"
TMPDIR="${TMPDIR:-${DEFAULT_TMPDIR}}"
SKIP_TORCH=0
SKIP_CUDA_MODULE=0

usage() {
  cat <<EOF
Usage:
  bash slurm/install_env.sh [options]

Options:
  --python <bin>            Python executable to use (default: python3)
  --venv <path>             Virtualenv path (default: ${DEFAULT_PACE_VENV_DIR})
  --torch-index-url <url>   PyTorch wheel index (default: ${TORCH_INDEX_URL})
  --pip-cache <path>        Pip cache directory (default: ${DEFAULT_PIP_CACHE_DIR})
  --skip-torch              Skip PyTorch installation
  --skip-cuda-module        Do not attempt 'module load cuda/12.6.1'
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:-}"
      shift 2
      ;;
    --pip-cache)
      PIP_CACHE_DIR="${2:-}"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --skip-cuda-module)
      SKIP_CUDA_MODULE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
PYTHON_BIN_PATH="$(command -v "${PYTHON_BIN}")"

cd "${REPO_ROOT}"

if [[ "${SKIP_CUDA_MODULE}" != "1" ]] && declare -F module >/dev/null 2>&1; then
  module load cuda/12.6.1 >/dev/null 2>&1 || true
fi

echo "[install] Repo root: ${REPO_ROOT}"
echo "[install] Creating/using virtualenv: ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
export PIP_CACHE_DIR TMPDIR
echo "[install] Pip cache: ${PIP_CACHE_DIR}"
echo "[install] TMPDIR: ${TMPDIR}"

env -u VIRTUAL_ENV PATH="$(dirname "${PYTHON_BIN_PATH}"):/usr/bin:/bin" \
  "${PYTHON_BIN_PATH}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_TORCH}" != "1" ]]; then
  python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==2.2.*"
fi

python -m pip install -r requirements-pip.txt

echo "[install] Running sanity import check"
python - <<'PY'
import importlib.util
import sys

modules = [
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "PIL",
    "torch",
    "gpytorch",
    "lightgbm",
]

missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    print("Missing modules:", ", ".join(missing))
    sys.exit(1)

import torch

print("Environment OK")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "[install] Done."
echo "Activate with: source ${VENV_DIR}/bin/activate"
