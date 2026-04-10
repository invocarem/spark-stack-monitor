# On Node 1, enter container and start server
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve Qwen/Qwen3.5-2B \
    --tensor-parallel-size 2 --max_model_len 2048'

