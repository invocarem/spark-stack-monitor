# vLLM 397B AutoRound without Ray: use --distributed-executor-backend mp.
#
# 1) Start the container (no Ray):
#    export VLLM_IMAGE=...   # your image, same as for head.sh
#    bash head_no_ray.sh     # on each host, or one host only for single-node
#
# 2) From this repo's UI/tools or manually, docker exec into the container and run
#    the inner bash -c block below (single-node: one machine, all GPUs).
#
# Multi-node mp (only if you split GPUs across hosts): on rank 0 run the same
# serve line plus e.g. --nnodes 2 --node-rank 0 --master-addr 192.168.100.11 --master-port 29501;
# on rank 1 add --node-rank 1 --headless. Match TP/PP to your hardware; see
# https://docs.vllm.ai/en/stable/serving/distributed_serving.html

set -euo pipefail

export VLLM_CONTAINER="${VLLM_CONTAINER:-vllm_node}"
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Single-node example (2 GPUs on one box): mp avoids Ray memory overhead.
docker exec -it "$VLLM_CONTAINER" /bin/bash -c "
  vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --served-model-name qwen3.5-397b \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    --host 0.0.0.0 \
    --load-format safetensors \
    --kv-cache-dtype fp8_e4m3 \
    --dtype auto \
    --enforce-eager \
    --enable-chunked-prefil \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-batched-tokens 4187 \
    --trust-remote-code \
    --mm-encoder-tp-mode data \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2"
