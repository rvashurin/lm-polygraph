#!/bin/bash
#SBATCH --job-name=direct_eval
#SBATCH --output=slurm_logs/direct.%A_%a.txt
#SBATCH --array=0-2                    # 0=Llama, 1=Qwen, 2=Falcon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-51
#SBATCH -p cscc-gpu-p
#SBATCH --time=14:00:00
#SBATCH --qos=cscc-gpu-qos

nvidia-smi
hostname
mkdir -p slurm_logs

cd ~/workspace/lm-polygraph
source .venv/bin/activate

# ==========================================
# CONFIGURATION
# ==========================================
ESTIMATOR_YAML="ugrip_benchmark_estimators_direct.yaml"
SAMPLE_SIZE=1000

MODELS=("ugrip_llama_instruct_vllm" "ugrip_qwen25_instruct_vllm" "ugrip_falcon_instruct_vllm")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Helper function to time specific datasets
run_dataset() {
    local DATASET_NAME=$1
    local DATASET_PATH=$2
    local SPLIT=$3

    echo ">>> Starting $DATASET_NAME for $MODEL..."
    local DS_START=$(date +%s)

    uv run --python 3.11 scripts/polygraph_eval \
        --config-dir=./examples/configs \
        --config-name=polygraph_eval_ugrip.yaml \
        model=$MODEL \
        dataset=$DATASET_PATH \
        generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
        estimators=$ESTIMATOR_YAML \
        subsample_eval_dataset=$SAMPLE_SIZE \
        batch_size=1 \
        eval_split=$SPLIT

    local DS_END=$(date +%s)
    local DS_DUR=$((DS_END - DS_START))
    echo "Finished $DATASET_NAME. Time taken: $((DS_DUR / 60))m $((DS_DUR % 60))s"
}

# ==========================================
# EVALUATION
# ==========================================

echo "Starting evaluation for $MODEL (Task ID: $SLURM_ARRAY_TASK_ID)"
MODEL_START=$(date +%s)

# Run datasets
run_dataset "GSM8K" "UGRIP-LM-Polygraph/gsm8k-direct" "test"
run_dataset "MMLU" "UGRIP-LM-Polygraph/mmlu-direct" "test"
run_dataset "MedMCQA" "UGRIP-LM-Polygraph/medmcqa-direct" "validation"

MODEL_END=$(date +%s)
TOTAL_DUR=$((MODEL_END - MODEL_START))

echo "======================================================"
echo "FINISHED ALL DATASETS FOR MODEL: $MODEL"
echo "Total Time taken: $((TOTAL_DUR / 60)) minutes and $((TOTAL_DUR % 60)) seconds."
echo "======================================================"