#!/bin/bash

# Configuration variables
MODEL="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
SERVED_MODEL_NAME="qwen3.5-397b"
CONTEXT_LENGTH=84000
MEM_FRACTION_STATIC=0.93
TENSOR_PARALLEL=2
HOST="0.0.0.0"
PORT=30000
ATTENTION_BACKEND="triton"
TOOL_CALL_PARSER="qwen3_coder"

SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --context-length ${CONTEXT_LENGTH} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --tp-size ${TENSOR_PARALLEL} \
    --host ${HOST} \
    --port ${PORT} \
    --enable-metrics \
    --attention-backend ${ATTENTION_BACKEND} \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-mamba-scheduler \
    --tool-call-parser ${TOOL_CALL_PARSER} \
    --reasoning-parser qwen3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-flashinfer-allreduce-fusion \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 3 \
    --enable-cache-report \
    --preferred-sampling-params '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --trust-remote-code
