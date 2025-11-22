#!/usr/bin/env bash
# getting encoder only adapter/LoRA weights from a full checkpoint
# usage details
#   scripts/get_adapters.sh CHECKPOINT OWSM_LANG ADAPTER_YAML OUT_DIR [OUT_NAME] [PRETRAINED_MODEL]
# example usage:
#   scripts/get_adapters.sh \
#     exp/adapter_gle/valid.total_count.best.pth \
#     gle \
#     config/finetune_adapter.yaml \
#     exports \
#     encoder_only_gle.pth \
#     espnet/owsm_v4_small_370M

set -euo pipefail

source espnet/tools/activate_python.sh

CHECKPOINT="${1:-}"
OWSM_LANG="${2:-}"
ADAPTER_YAML="${3:-}"
OUT_DIR="${4:-}"
OUT_NAME="${5:-}"
PRETRAINED_MODEL="${6:-espnet/owsm_v4_small_370M}"

if [ -z "$CHECKPOINT" ] || [ -z "$OWSM_LANG" ] || [ -z "$ADAPTER_YAML" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 CHECKPOINT OWSM_LANG ADAPTER_YAML OUT_DIR [OUT_NAME] [PRETRAINED_MODEL]"
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "[$(date)] Starting adapter export"
echo "Checkpoint      : ${CHECKPOINT}"
echo "OWSM language   : ${OWSM_LANG}"
echo "Adapter config  : ${ADAPTER_YAML}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Out dir         : ${OUT_DIR}"
if [[ -n "${OUT_NAME}" ]]; then
  echo "Out name        : ${OUT_NAME}"
fi
echo "----------------------------------------"

CMD=( python3 scripts/get_adapters.py
  --checkpoint "${CHECKPOINT}"
  --owsm_lang "${OWSM_LANG}"
  --adapter_config "${ADAPTER_YAML}"
  --pretrained_model "${PRETRAINED_MODEL}"
  --out_dir "${OUT_DIR}"
)

# Append optional out_name if provided
if [[ -n "${OUT_NAME}" ]]; then
  CMD+=( --out_name "${OUT_NAME}" )
fi

echo "[cmd] ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Export completed"
