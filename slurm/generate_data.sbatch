#!/bin/bash
#SBATCH --job-name=pso_generate_data
#SBATCH -A gts-sn73
#SBATCH --qos=inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/pace_common.sh"

RUNS="${RUNS:-500}"
N_ITERS="${ITERS:-200}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
DEVICE="${PSO_DEVICE:-cuda:0}"
OUT_PATH="${OUT_PATH:-${PACE_ARTIFACT_ROOT}/training_data.npz}"
FUNCTIONS="${FUNCTIONS:-}"
DIMS="${DIMS:-}"

if [[ -f "${OUT_PATH}" && "${FORCE_REGEN:-0}" != "1" ]]; then
  echo "[$(date '+%F %T')] SKIP: ${OUT_PATH} already exists"
  exit 0
fi

echo "Runs: ${RUNS}"
echo "Iters: ${N_ITERS}"
echo "Sample every: ${SAMPLE_EVERY}"
echo "Device: ${DEVICE}"
echo "Output: ${OUT_PATH}"
echo "Functions: ${FUNCTIONS:-<default>}"
echo "Dims: ${DIMS:-<default>}"
nvidia-smi || true

cmd=(
  "${VENV_PY}" generate_training_data.py
  --runs "${RUNS}"
  --iters "${N_ITERS}"
  --sample-every "${SAMPLE_EVERY}"
  --device "${DEVICE}"
  --out "${OUT_PATH}"
)

if [[ -n "${FUNCTIONS}" ]]; then
  cmd+=(--functions "${FUNCTIONS}")
fi

if [[ -n "${DIMS}" ]]; then
  cmd+=(--dims "${DIMS}")
fi

"${cmd[@]}"
