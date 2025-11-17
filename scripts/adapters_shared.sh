#!/usr/bin/env bash

source espnet/tools/activate_python.sh

EXP_DIR=exp/adapter_shared
STATS_DIR=${EXP_DIR}/stats

mkdir -p "${EXP_DIR}" "${STATS_DIR}"

echo "[$(date)] Starting shared adapter training (hun, rus, cym, ita, por)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/adapters_shared.py \
  --exp_dir ${EXP_DIR} \
  --stats_dir ${STATS_DIR} \
  --finetune_config config/finetune_adapter_shared.yaml \
  --pretrained_model espnet/owsm_v4_small_370M

echo "[$(date)] Shared finetuning completed"
