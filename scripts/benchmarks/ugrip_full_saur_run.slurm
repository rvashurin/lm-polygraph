#!/bin/bash
#SBATCH --job-name=full_saur           # Job name updated
#SBATCH --output=slurm_logs/output.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                      # Run all processes on a single node  
#SBATCH --ntasks=1                     # Run on a single CPU
#SBATCH --mem=100G                     # Total RAM to be used
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --gres=gpu:2                   # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                  # Use the gpu partition
#SBATCH --time=04:00:00                # Short time limit for testing
#SBATCH --qos=cscc-gpu-qos             # To enable the use of up to 8 GPUs

nvidia-smi
hostname

mkdir -p slurm_logs
cd ~/workspace/lm-polygraph

source .venv/bin/activate

# Define variables for easy switching later
MODEL="ugrip_gemma_instruct_vllm" 
ESTIMATOR_YAML="saur_estimator.yaml" # Pointing to the new SAUR config
SAMPLE_SIZE=-1 # The "small set of data" for testing

echo "Starting test run for Model: $MODEL, Estimator: $ESTIMATOR_YAML, Sample Size: $SAMPLE_SIZE"

# 1. GSM8K (uses 'test' split)
echo "Running GSM8K..."
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

# 2. MMLU (uses 'test' split)
echo "Running MMLU..."
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip.yaml \
  model=$MODEL \
  dataset=UGRIP-LM-Polygraph/mmlu-reasoning \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=$ESTIMATOR_YAML \
  subsample_eval_dataset=$SAMPLE_SIZE \
  batch_size=1 \
  eval_split=test

# 3. MedMCQA (uses 'validation' split based on your script history)
echo "Running MedMCQA..."
uv run --python 3.11 scripts/polygraph_eval \
  --config-dir=./examples/configs \
  --config-name=polygraph_eval_ugrip.yaml \
  model=$MODEL \
  dataset=UGRIP-LM-Polygraph/medmcqa-reasoning \
  generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
  estimators=$ESTIMATOR_YAML \
  subsample_eval_dataset=$SAMPLE_SIZE \
  batch_size=1 \
  eval_split=validation

echo "All test jobs finished at $(date)"