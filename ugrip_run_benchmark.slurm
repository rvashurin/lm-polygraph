#!/bin/bash

#SBATCH --job-name=ugrip_polygraph_benchmarks_p4d
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Create directory for slurm logs if it doesn't 
# exist yet
mkdir -p slurm_logs

echo "Setting up conda environment..."
source activate base
conda activate reasoning_uq

echo "Starting script..."
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml model=ugrip_gemma_instruct
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml hydra.mode=MULTIRUN estimators=ugrip_benchmark_estimators_lite model=ugrip_llama_instruct,ugrip_gemma_instruct subsample_eval_dataset=10 max_new_tokens=2 dataset=UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning

# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml hydra.mode=MULTIRUN estimators=ugrip_benchmark_estimators_lite model=ugrip_llama_instruct,ugrip_gemma_instruct subsample_eval_dataset=10 max_new_tokens=2 dataset=UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning

# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml model=ugrip_gemma_instruct subsample_eval_dataset=2 max_new_tokens=2
# scripts/polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_ugrip.yaml hydra.mode=MULTIRUN model=ugrip_gemma_instruct subsample_eval_dataset=1 dataset=UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/medmcqa-direct max_new_tokens=3


echo "Starting Llama-3.1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ./scripts/polygraph_eval \
    --config-dir ./examples/configs \
    --config-name polygraph_eval_ugrip \
    --multirun \
    model=ugrip_llama_instruct \
    dataset=UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning &

echo "Starting Gemma-2 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 ./scripts/polygraph_eval \
    --config-name polygraph_eval_ugrip \
    --config-dir ./examples/configs \
    --multirun \
    model=ugrip_gemma_instruct \
    dataset=UGRIP-LM-Polygraph/gsm8k-direct,UGRIP-LM-Polygraph/gsm8k-reasoning,UGRIP-LM-Polygraph/medmcqa-direct,UGRIP-LM-Polygraph/medmcqa-reasoning,UGRIP-LM-Polygraph/mmlu-direct,UGRIP-LM-Polygraph/mmlu-reasoning &


echo "Waiting for all benchmark runs to complete..."
wait 

echo "All jobs finished at $(date)"