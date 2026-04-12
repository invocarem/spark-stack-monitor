# On Node 1, enter container and start server
export VLLM_CONTAINER="vllm_node_tf5"

# On spark1, when launching vLLM
docker exec -it $VLLM_CONTAINER /bin/bash -c "
 vllm serve Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --host 0.0.0.0 \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray"
