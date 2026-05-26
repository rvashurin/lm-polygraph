#!/bin/bash
#SBATCH --job-name=test_msp            
#SBATCH --output=slurm_logs/output.%A_%a.txt   
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --mem=200G                     
#SBATCH --cpus-per-task=10             
#SBATCH --gres=gpu:4                   
#SBATCH -p cscc-gpu-p                  
#SBATCH --time=00:20:00                
#SBATCH --qos=cscc-gpu-qos             

nvidia-smi
hostname

mkdir -p slurm_logs
cd ~/workspace/lm-polygraph

source .venv/bin/activate

MODEL="ugrip_gemma_instruct_vllm" 
ESTIMATOR_YAML="msp_estimator.yaml" # Using the isolated MSP config
SAMPLE_SIZE=1 # Sanity check size

echo "Starting MSP Sanity Check | Model: $MODEL | Size: $SAMPLE_SIZE"

# 1. GSM8K
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

# 2. MMLU
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

# 3. MedMCQA
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

echo "Sanity check finished at $(date)"