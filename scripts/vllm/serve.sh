#!/usr/bin/env bash
# Editable vLLM launch script (repo → /workspace/scripts/vllm in the test container).
# Run from the monitor Tools entry "vLLM: launch serve.sh (detached)".

set -euo pipefail

MODEL="${VLLM_MODEL:-Intel/Qwen3.5-122B-A10B-int4-AutoRound}"

exec vllm serve "${MODEL}" \
  --served-model-name qwen3.5-122b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --chat-template unsloth.jinja \
  -tp 1 \
  --distributed-executor-backend ray

