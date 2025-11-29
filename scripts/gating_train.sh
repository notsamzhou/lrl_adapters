#!/usr/bin/env bash
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
set -x

# --- Absolute paths you own ---
ESPNET_ROOT="/jet/home/ayang4/espnet/espnet"
PYTHON_BIN="/jet/home/ayang4/espnet/espnet/tools/miniconda/envs/espnet311/bin/python"

# Use ESPnet sources and avoid user-site shadowing
export PYTHONPATH="${ESPNET_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# Force NumPy 1.26 override from project space FIRST on sys.path
export PYDEPS_DIR="/ocean/projects/cis250187p/ayang4/pydeps/numpy126"
export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH}"

# Activate the conda env
source /jet/home/ayang4/espnet/espnet/tools/activate_python.sh

# Central HF caches you own (avoid /var/spool perms)
export HF_CACHE_BASE="/ocean/projects/cis250187p/ayang4/hf_cache"
export XDG_CACHE_HOME="${HF_CACHE_BASE}"
export HF_HOME="${HF_CACHE_BASE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}"
export HF_HUB_CACHE="${HF_CACHE_BASE}"
export TRANSFORMERS_CACHE="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export ESPNET_MODEL_ZOO_CACHE_DIR="${HF_CACHE_BASE}/espnet_zoo"
export TMPDIR="${HF_CACHE_BASE}/tmp"
export HF_HUB_DISABLE_TELEMETRY=1
export HUGGINGFACE_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_CACHE_BASE" "$HF_DATASETS_CACHE" "$ESPNET_MODEL_ZOO_CACHE_DIR" "$TMPDIR"

# --- Args ---
TARGET_FLEURS="${1:-}"; shift || true
TARGET_OWSM="${1:-}"; shift || true
LAYER_IDS="${1:-0,4,8,12,16,20}"; shift || true
SAVE_DIR="${1:-exp/gate_target}"; shift || true

# Everything remaining is adapter paths
ADAPTER_BANK=("$@")

if [[ -z "${TARGET_FLEURS}" || -z "${TARGET_OWSM}" || ${#ADAPTER_BANK[@]} -eq 0 ]]; then
  echo "Usage: $0 \"ga_ie ...\" \"gle ...\" \"0,4,8\" \"exp/gate_target\" <adapter1.pt> <adapter2.pt> ..."
  exit 2
fi

echo "[$(date)] Starting gating training"
echo "[env] python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, numpy as np
print("[env] Python:", sys.version.split()[0])
print("[env] NumPy:", np.__version__, "@", np.__file__)
import torch
print("[env] Torch:", torch.__version__)
import espnet
print("[env] ESPnet:", getattr(espnet, "__version__", "unknown"))
print("[env] sys.path[:3]:", sys.path[:3])
PY

# Run training with the exact interpreter
"$PYTHON_BIN" scripts/gating_train.py \
  --pretrained_model espnet/owsm_v4_small_370M \
  --target_fleurs ${TARGET_FLEURS} \
  --target_owsm   ${TARGET_OWSM} \
  --adapter_bank  "${ADAPTER_BANK[@]}" \
  --layers ${LAYER_IDS} \
  --topk 2 \
  --tau 1.0 \
  --batch_size 1 \
  --epochs 2 \
  --lr 5e-4 \
  --max_seconds 12 \
  --save_dir "${SAVE_DIR}"

echo "[$(date)] Done. Gate checkpoints in ${SAVE_DIR}/"
