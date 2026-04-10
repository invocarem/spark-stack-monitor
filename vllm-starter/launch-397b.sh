# On HEAD node (192.168.100.11)
export HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -1)

docker exec -it $HEAD_CONTAINER /bin/bash -c "
  export NCCL_SOCKET_IFNAME=enp1s0f0np0
  export GLOO_SOCKET_IFNAME=enp1s0f0np0
  export TP_SOCKET_IFNAME=enp1s0f0np0
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_USE_FLASHINFER_MOE_FP16=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export OMP_NUM_THREADS=4

  vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --enable-prefix-caching
"
