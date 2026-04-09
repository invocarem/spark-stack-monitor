#!/usr/bin/env bash
# Editable vLLM launch script (repo → /workspace/scripts/vllm in the test container).
# Run from the monitor Tools entry "vLLM: launch serve.sh (detached)".

set -euo pipefail

exec vllm serve Qwen/Qwen3.5-35B-A3B \
    --served-model-name qwen3.5-35b \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --load-format fastsafetensors \
    --attention-backend flashinfer \
    --max-model-len 262144 \
    --reasoning-parser qwen3