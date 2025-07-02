#!/bin/bash 

# Dataset configs
DATASETS="UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning"
REASONING_DATASETS="UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-reasoning"

# Model configs
MODELS="ugrip_gemma_instruct,ugrip_llama_instruct"
LLAMA_MODEL="ugrip_llama_instruct"
GEMMA_MODEL="ugrip_gemma_instruct"

# Config args for every run
CONFIG_ARGS="--config-dir ./examples/configs --config-name polygraph_eval_ugrip hydra.mode=MULTIRUN"
CONFIG_ARGS_NO_MULTI="--config-dir ./examples/configs --config-name polygraph_eval_ugrip"

# Logging output  
LOG_FILE="hyperpod/benchmark_output_$(date +"%m-%d-%y_%H-%M").log"

# SUBSAMPLE_DATASET_SIZE (set -1 to ignore)
SUBSAMPLE_EVAL_DATASET=10

mkdir -p hyperpod
exec &> >(tee "${LOG_FILE}")

# Activate the conda environment
echo "Setting up reasoning_uq conda environment"
source activate base 
conda activate reasoning_uq 

trap 'kill $BGPID; exit' INT


START_TIME=$(date +%s)

echo "Starting benchmarking jobs at $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
    $CONFIG_ARGS \
    model=$GEMMA_MODEL \
    dataset=UGRIP-LM-Polygraph/mmlu-reasoning \
    subsample_eval_dataset=10 &


# echo "GEMMA ON GSM8K-REASONING"
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
#     $CONFIG_ARGS_NO_MULTI model=$GEMMA_MODEL dataset=UGRIP-LM-Polygraph/gsm8k-reasoning \
#     subsample_eval_dataset=$SUBSAMPLE_EVAL_DATASET &
# BGPID=$!
# wait

# echo "GEMMA ON MEDMCQA-REASONING"
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
#     $CONFIG_ARGS_NO_MULTI model=$GEMMA_MODEL dataset=UGRIP-LM-Polygraph/medmcqa-reasoning \
#     subsample_eval_dataset=$SUBSAMPLE_EVAL_DATASET &
# BGPID=$!
# wait

# echo "GEMMA ON MMLU-REASONING"
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
#     $CONFIG_ARGS_NO_MULTI model=$GEMMA_MODEL dataset=UGRIP-LM-Polygraph/mmlu-reasoning \
#     subsample_eval_dataset=$SUBSAMPLE_EVAL_DATASET &
# BGPID=$!

wait 

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

echo "All benchmarking jobs finished at $(date)" 