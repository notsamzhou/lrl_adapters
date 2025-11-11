#!/usr/bin/env bash

source espnet/tools/activate_python.sh

FLEURS_LANG="$1"
OWSM_LANG="$2"


EXP_DIR=exp/adapter_${OWSM_LANG}
STATS_DIR=${EXP_DIR}/stats

mkdir -p "${EXP_DIR}" "${STATS_DIR}"


echo "[$(date)] Starting adapter training for FLEURS=${FLEURS_LANG}, OWSM_LANG=${OWSM_LANG}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/adapters.py \
  --fleurs_lang ${FLEURS_LANG} \
  --owsm_lang ${OWSM_LANG} \
  --exp_dir ${EXP_DIR} \
  --stats_dir ${STATS_DIR} \
  --finetune_config config/finetune_adapter.yaml \
  --pretrained_model espnet/owsm_v4_base_102M

echo "[$(date)] Finetuning completed for FLEURS=${FLEURS_LANG}, OWSM_LANG=${OWSM_LANG}"