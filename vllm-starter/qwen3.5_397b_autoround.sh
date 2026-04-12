# Run from the monitor Tools entry "vLLM: launch serve.sh (detached)".
#
# Ray uses extra memory (object store / workers). For 397B AutoRound if the load
# fails with Ray, use head_no_ray.sh + qwen3.5_397b_autoround_mp.sh instead
# (--distributed-executor-backend mp).

set -euo pipefail
# On Node 1, enter container and start server
export VLLM_CONTAINER="vllm_node"
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_MARLIN_USE_ATOMIC_ADD=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# On spark1, when launching vLLM
docker exec -it $VLLM_CONTAINER /bin/bash -c "
  vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --served-model-name qwen3.5-397b \
        --max-model-len 4096 \
        --max-num-seqs 1 \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --port 8000 \
        --host 0.0.0.0 \
        --load-format safetensors\
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
        --distributed-executor-backend ray \
        --tensor-parallel-size 2"

