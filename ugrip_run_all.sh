#!/bin/bash 

# Dataset configs
DATASETS="UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning"
REASONING_DATASETS="UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-reasoning"

# Model configs
MODELS="ugrip_llama_instruct,ugrip_gemma_instruct"
LLAMA_MODEL="ugrip_llama_instruct"
GEMMA_MODEL="ugrip_gemma_instruct"

# Config args for every run
CONFIG_ARGS="--config-dir ./examples/configs --config-name polygraph_eval_ugrip hydra.mode=MULTIRUN"

# Logging output  
LOG_FILE="hyperpod/benchmark_output_$(date).log"

mkdir -p hyperpod
exec &> >(tee "${LOG_FILE}")

# Activate the conda environment
echo "Setting up reasoning_uq conda environment"
source activate base 
conda activate reasoning_uq 

echo "Starting benchmarking jobs at $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
    $CONFIG_ARGS \
    model=$MODELS \
    dataset=$REASONING_DATASETS \
    subsample_eval_dataset=20 &

echo "Waiting for completion"
wait 

echo "All benchmarking jobs finished at $(date)"