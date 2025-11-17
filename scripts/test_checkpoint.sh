#!/usr/bin/env bash

source espnet/tools/activate_python.sh

CHECKPOINT="$1"
TEST_FLEURS="$2"
TEST_OWSM="$3"
NUM_SAMPLES="${4:-}"  # Optional

OUTPUT_DIR="test_results"
mkdir -p "${OUTPUT_DIR}"

MODEL_NAME=$(basename $(dirname ${CHECKPOINT}))

OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_on_${TEST_OWSM}.txt"

echo "[$(date)] Starting model test"
echo "Checkpoint: ${CHECKPOINT}"
echo "Testing on: ${TEST_FLEURS} (${TEST_OWSM})"
echo "Output file: ${OUTPUT_FILE}"
echo "----------------------------------------"

CMD="python3 scripts/test_checkpoint.py \
  --checkpoint ${CHECKPOINT} \
  --fleurs_lang ${TEST_FLEURS} \
  --owsm_lang ${TEST_OWSM} \
  --output_file ${OUTPUT_FILE}"

# Add num_samples if provided
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples ${NUM_SAMPLES}"
fi

eval $CMD

echo "[$(date)] Evaluation completed"
echo "Results saved to: ${OUTPUT_FILE}"
