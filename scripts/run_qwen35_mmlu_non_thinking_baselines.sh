#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HYDRA_CONFIG="$PWD/examples/configs/polygraph_eval_mmlu_qwen3_5_27b_non_thinking_baselines.yaml" "${LM_POLYGRAPH_UV_ENV:-$PWD/.venv-qwen35}/bin/python" "$PWD/scripts/polygraph_eval" cache_path="${LM_POLYGRAPH_CACHE_PATH:-$PWD/workdir/output}" save_path="${LM_POLYGRAPH_SAVE_PATH:-$PWD/workdir/qwen35_mmlu_non_thinking_baselines}" "$@"
