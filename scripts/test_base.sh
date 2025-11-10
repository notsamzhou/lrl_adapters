#!/usr/bin/env bash

source espnet/tools/activate_python.sh

FLEURS_LANG="$1"
OWSM_LANG="$2"
NUM_SAMPLES="${3:-}"  # Optional

OUTPUT_DIR="test_results"
mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/base_model_${OWSM_LANG}.txt"

echo "[$(date)] Starting base model evaluation"
echo "Language: ${FLEURS_LANG} (${OWSM_LANG})"
echo "Output file: ${OUTPUT_FILE}"
echo "----------------------------------------"

CMD="python3 scripts/test_base_model.py \
  --fleurs_lang ${FLEURS_LANG} \
  --owsm_lang ${OWSM_LANG} \
  --output_file ${OUTPUT_FILE}"

# Add num_samples if provided
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples ${NUM_SAMPLES}"
fi


eval $CMD

echo "[$(date)] Evaluation completed"
echo "Results saved to: ${OUTPUT_FILE}"
