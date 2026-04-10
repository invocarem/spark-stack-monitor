
# On spark1, when launching vLLM
  vllm serve Qwen/Qwen3.5-2B \
    --served-model-name qwen3.5-2b \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.4 \
    --distributed-executor-backend ray

