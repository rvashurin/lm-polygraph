#!/bin/bash 

source activate base 
conda activate reasoning_uq 
DATASETS="UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning"
# MODELS="ugrip_llama_instruct,ugrip_gemma_instruct"
MODELS="ugrip_llama_instruct"
CONFIG_ARGS="--config-dir ./examples/configs --config-name polygraph_eval_ugrip hydra.mode=MULTIRUN"

echo "Starting script"
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/polygraph_eval \
    $CONFIG_ARGS \
    model=$MODELS \
    dataset=$DATASETS \
    subsample_eval_dataset=10 \
    max_new_tokens=50 &

echo "Waiting for completion"
wait 

echo "All jobs finished at $(date)"