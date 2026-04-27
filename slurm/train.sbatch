#!/bin/bash
#SBATCH --job-name=pso_train_classifier
#SBATCH -A gts-sn73
#SBATCH --qos=embers
#SBATCH -p cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/pace_common.sh"

TRAIN_DATA="${TRAIN_DATA:-${PACE_ARTIFACT_ROOT}/training_data.npz}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-${PACE_ARTIFACT_ROOT}/stuck_classifier.lgb}"
LGB_DEVICE="${LGB_DEVICE:-cpu}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "ERROR: training data not found at ${TRAIN_DATA}" >&2
  exit 1
fi

if [[ -f "${CLASSIFIER_PATH}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
  echo "[$(date '+%F %T')] SKIP: ${CLASSIFIER_PATH} already exists"
  exit 0
fi

echo "Training data: ${TRAIN_DATA}"
echo "Classifier path: ${CLASSIFIER_PATH}"
echo "LightGBM device request: ${LGB_DEVICE}"

"${VENV_PY}" train_classifier.py \
  --data "${TRAIN_DATA}" \
  --out "${CLASSIFIER_PATH}" \
  --device "${LGB_DEVICE}"
