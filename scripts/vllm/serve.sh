#!/usr/bin/env bash
# Editable vLLM launch script (repo → /workspace/scripts/vllm in the test container).
# Run from the monitor Tools entry "vLLM: launch serve.sh (detached)".

set -euo pipefail
MODEL="Qwen/Qwen3.5-2B"
SERVED_MODEL_NAME="qwen3.5-2b"
exec vllm serve ${MODEL} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --gpu-memory-utilization 0.35 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --load-format fastsafetensors \
    --attention-backend flashinfer \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --distributed-executor-backend  ray