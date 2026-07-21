#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_DIR="${LM_POLYGRAPH_UV_ENV:-$PWD/.venv-qwen35}"
PYTHON_BIN="${LM_POLYGRAPH_PYTHON:-python3.12}"
TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"

if ! command -v uv >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "${PYTHON_BIN}" "${ENV_DIR}"

UV_PIP_ARGS=(--python "${ENV_DIR}/bin/python")
if [ -n "${TORCH_BACKEND}" ]; then
  UV_PIP_ARGS+=(--torch-backend "${TORCH_BACKEND}")
fi

uv pip install "${UV_PIP_ARGS[@]}" -e .
uv pip install "${UV_PIP_ARGS[@]}" --upgrade "transformers>=4.57.0" "accelerate>=1.8.0" "huggingface-hub>=0.33.0"
