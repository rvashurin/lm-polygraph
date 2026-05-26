#!/bin/bash
#SBATCH --job-name=saur_fast
#SBATCH --output=slurm_logs/fast.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G                      # Bumped to support Ray
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:2                   
#SBATCH -p cscc-gpu-p
#SBATCH --time=00:15:00                

mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
source .venv/bin/activate

MODEL="ugrip_llama_instruct_vllm" 
ESTIMATOR_YAML="saur_estimator_llama.yaml" 
SAMPLE_SIZE=2 # 2 samples just to prove the SAUR math executes

echo "Running Fast Test on 2 GPUs..."
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip.yaml \
  model=$MODEL \
  dataset=UGRIP-LM-Polygraph/gsm8k-reasoning \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=$ESTIMATOR_YAML \
  subsample_eval_dataset=$SAMPLE_SIZE \
  batch_size=1 \
  eval_split=test