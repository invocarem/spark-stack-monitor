# On Node 1, enter container and start server
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')

# On spark1, when launching vLLM
#
#

docker exec -it $VLLM_CONTAINER /bin/bash -c "
  export TOKENIZERS_PARALLELISM=false && \
  export VLLM_USE_DEEP_GEMM=0 && \
  export VLLM_USE_FLASHINFER_MOE_FP16=1 && \
  vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --tokenizer Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --load-format fastsafetensors \
    --enable-prefix-caching \
    --distributed-executor-backend ray"

