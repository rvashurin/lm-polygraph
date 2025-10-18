#!/bin/bash

# Activate uv venv if it isn't already
source .venv/bin/activate 

# Config settings
MODEL="ugrip_llama_instruct_vllm"
DATASET="UGRIP-LM-Polygraph/gsm8k-reasoning"
SAMPLE_SIZE=2
TOTAL_START_TIME=$SECONDS
LOG_FILE="ugrip_logs/$(date +%Y%m%d)/run_$(date +%H%M%S).log"

# if log dir doesn't exist, create it
mkdir -p "$(dirname "$LOG_FILE")"

{
  echo "Running on host: $(hostname)"
  echo "Model: $MODEL, Dataset: $DATASET, Sample Size: $SAMPLE_SIZE"

  echo "========================================================"
  echo "Slicing Debugging"
  echo "Host: $(hostname)"
  echo "Model: $MODEL, Dataset: $DATASET, Sample Size: $SAMPLE_SIZE"
  echo "Log file: $LOG_FILE"
  echo "========================================================"
  echo ""

  echo "--- Running REASONING test ---"
  scripts/polygraph_eval \
    --config-dir=./examples/configs \
    --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml \
    model=$MODEL \
    dataset=$DATASET \
    subsample_eval_dataset=$SAMPLE_SIZE

  echo "--- Finished REASONING test ---"
  echo ""

  # --- Test 2: Answer Slicing ---
  echo "--- Running ANSWER test ---"
  scripts/polygraph_eval \
    --config-dir=./examples/configs \
    --config-name=polygraph_eval_ugrip_segmentation_answer.yaml \
    model=$MODEL \
    dataset=$DATASET \
    subsample_eval_dataset=$SAMPLE_SIZE
    
  echo "--- Finished ANSWER test ---"
  echo "========================================================"
} 2>&1 | tee "$LOG_FILE"

echo "Finished script, all output saved in $LOG_FILE"
