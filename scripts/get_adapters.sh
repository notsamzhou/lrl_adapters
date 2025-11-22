#!/usr/bin/env bash
# getting only adapter/LoRA weights from a full checkpoint
# usage details:
#   scripts/get_adapters.sh CHECKPOINT OWSM_LANG ADAPTER_YAML OUTPUT_PT [PRETRAINED_MODEL]
# example run command:
#   scripts/get_adapters.sh \
#     exp/adapter_gle/valid.total_count.best.pth \
#     gle \
#     config/finetune_adapter.yaml \
#     exports/gle_adapters.pt \
#     espnet/owsm_v4_small_370M

set -euo pipefail

source espnet/tools/activate_python.sh

CHECKPOINT="${1:-}"
OWSM_LANG="${2:-}"
ADAPTER_YAML="${3:-}"
OUTPUT_PT="${4:-}"
PRETRAINED_MODEL="${5:-espnet/owsm_v4_small_370M}"

if [ -z "$CHECKPOINT" ] || [ -z "$OWSM_LANG" ] || [ -z "$ADAPTER_YAML" ] || [ -z "$OUTPUT_PT" ]; then
  echo "Usage: $0 CHECKPOINT OWSM_LANG ADAPTER_YAML OUTPUT_PT [PRETRAINED_MODEL]"
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PT")"

echo "[$(date)] Starting adapter export"
echo "Checkpoint      : ${CHECKPOINT}"
echo "OWSM language   : ${OWSM_LANG}"
echo "Adapter config  : ${ADAPTER_YAML}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Output file     : ${OUTPUT_PT}"
echo "----------------------------------------"

CMD="python3 scripts/get_adapters.py \
  --checkpoint ${CHECKPOINT} \
  --owsm_lang ${OWSM_LANG} \
  --adapter_config ${ADAPTER_YAML} \
  --pretrained_model ${PRETRAINED_MODEL} \
  --output ${OUTPUT_PT}"

echo "[cmd] $CMD"
eval "$CMD"

echo "[$(date)] Export completed"
echo "Saved to: ${OUTPUT_PT}"
