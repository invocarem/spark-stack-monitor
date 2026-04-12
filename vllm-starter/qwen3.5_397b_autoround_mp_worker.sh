# Multi-node mp, rank 1 (worker): run on the worker host after worker_no_ray.sh is up.
#
# Prereq: bash worker_no_ray.sh on this machine (container vllm_node_tf5_worker by default).
# Start rank 0 (qwen3.5_397b_autoround_mp_head.sh) on the head host before or closely after.
#
# --headless: worker does not bind the OpenAI HTTP port; API is on the head only.
#
# Override: MASTER_ADDR, MASTER_PORT, NNODES, VLLM_CONTAINER

set -euo pipefail

export VLLM_CONTAINER="${VLLM_CONTAINER:-vllm_node_tf5_worker}"
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MASTER_ADDR="${MASTER_ADDR:-192.168.100.11}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-2}"

docker exec -it \
  -e MASTER_ADDR="$MASTER_ADDR" \
  -e MASTER_PORT="$MASTER_PORT" \
  "$VLLM_CONTAINER" /bin/bash -c "
  vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --served-model-name qwen3.5-397b \
    --max-model-len 16384 \
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
    --nnodes ${NNODES} \
    --node-rank 1 \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    --tensor-parallel-size 2 \
    --headless"
