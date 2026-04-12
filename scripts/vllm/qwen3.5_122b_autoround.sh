#!/usr/bin/env bash
# Editable vLLM launch script (repo → /workspace/scripts/vllm in the test container).
# Run from the monitor Tools entry "vLLM: launch serve.sh (detached)".

set -euo pipefail

exec vllm serve Intel/Qwen3.5-122B-A10B-int4-AutoRound \
    --served-model-name qwen3.5-122b \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --gpu-memory-utilization 0.88\
    --load-format fastsafetensors \
    --attention-backend flashinfer \
    --max-model-len 32768 \
    --enforce-eager \
    --max-num-seqs 2 \
    --reasoning-parser qwen3 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray
