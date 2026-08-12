#!/bin/bash
#SBATCH --job-name=ugrip_reasoning   # Job name
#SBATCH --output=output.%A_%a.txt    # Standard output and error log
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --mem=200G                   # Total RAM to be used
#SBATCH --cpus-per-task=10           # Number of CPU cores
#SBATCH --gres=gpu:4                 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                # Use the gpu partition
#SBATCH --time=24:00:00              # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos           # To enable the use of up to 8 GPUs


nvidia-smi
hostname


mkdir -p slurm_logs
cd ~/workspace/lm-polygraph
source .venv/bin/activate


# Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / Falcon3-7B-Instruct on the
# *-reasoning datasets. For each model+dataset pair we produce three .man
# files corresponding to three UQ slice variants:
#
#   full      -> polygraph_eval_ugrip.yaml                       (slicing_target=null)
#   reasoning -> polygraph_eval_ugrip_segmentation_reasoning.yaml (slicing_target=reasoning,
#                strips leading '### Reasoning:' label from the slice)
#   answer    -> polygraph_eval_ugrip_segmentation_answer.yaml    (slicing_target=answer,
#                slice already starts past '### Answer:')
#
# Greedy decoding is deterministic given identical model/prompt/seed, so the
# greedy_texts and extracted answers are identical across the three runs; only
# the UQ values change because the underlying log-probs/attentions/embeddings
# are restricted to a different token range.


for MODEL in ugrip_llama_instruct ugrip_qwen25_instruct ugrip_falcon3_instruct; do
  for DATASET in UGRIP-LM-Polygraph/gsm8k-reasoning UGRIP-LM-Polygraph/mmlu-reasoning UGRIP-LM-Polygraph/medmcqa-reasoning; do
    SPLIT=test
    FN=process_output_mcq
    [[ "$DATASET" == *medmcqa* ]] && SPLIT=validation
    [[ "$DATASET" == *gsm8k* ]] && FN=process_output_number

    # 1) full output
    uv run --python 3.11 scripts/polygraph_eval \
      --config-dir=./examples/configs \
      --config-name=polygraph_eval_ugrip.yaml \
      model=$MODEL \
      dataset=$DATASET \
      generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
      estimators=ugrip_benchmark_estimators_lite.yaml \
      process_output_fn.fn_name=$FN \
      use_claim_ue=false \
      +output_attentions=false \
      subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT

    # 2) reasoning slice (label '### Reasoning:' stripped from UQ range)
    uv run --python 3.11 scripts/polygraph_eval \
      --config-dir=./examples/configs \
      --config-name=polygraph_eval_ugrip_segmentation_reasoning.yaml \
      model=$MODEL \
      dataset=$DATASET \
      generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
      estimators=ugrip_benchmark_estimators_lite.yaml \
      process_output_fn.fn_name=$FN \
      use_claim_ue=false \
      +output_attentions=false \
      subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT

    # 3) answer slice (slice already starts past '### Answer:')
    uv run --python 3.11 scripts/polygraph_eval \
      --config-dir=./examples/configs \
      --config-name=polygraph_eval_ugrip_segmentation_answer.yaml \
      model=$MODEL \
      dataset=$DATASET \
      generation_metrics=ugrip_benchmark_generation_metrics_acc.yaml \
      estimators=ugrip_benchmark_estimators_lite.yaml \
      process_output_fn.fn_name=$FN \
      use_claim_ue=false \
      +output_attentions=false \
      subsample_eval_dataset=1000 batch_size=1 eval_split=$SPLIT
  done
done


echo "Waiting for all benchmark runs to complete..."
wait

echo "All jobs finished at $(date)"
