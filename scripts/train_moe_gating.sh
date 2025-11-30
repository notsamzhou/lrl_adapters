#!/usr/bin/env bash

source espnet/tools/activate_python.sh

FLEURS_LANG="$1"
OWSM_LANG="$2"

if [ -z "$FLEURS_LANG" ] || [ -z "$OWSM_LANG" ]; then
    echo "Usage: $0 <fleurs_lang> <owsm_lang>"
    echo "Example: $0 es_419 spa"
    exit 1
fi

EXP_DIR=exp/moe_gating_${OWSM_LANG}
STATS_DIR=${EXP_DIR}/stats

mkdir -p "${EXP_DIR}" "${STATS_DIR}"

echo "[$(date)] Starting MoE gating training for target=${OWSM_LANG}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/train_moe_gating.py \
  --fleurs_lang ${FLEURS_LANG} \
  --target_lang ${OWSM_LANG} \
  --exp_dir ${EXP_DIR} \
  --stats_dir ${STATS_DIR} \
  --finetune_config config/finetune_moe_gating.yaml \
  --pretrained_model espnet/owsm_v4_small_370M

echo "[$(date)] MoE gating training completed for target=${OWSM_LANG}"
