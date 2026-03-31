exec vllm serve \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
	--served-model-name qwen3-next-80b \
	--tool-call-parser hermes  \
    --enable-auto-tool-choice   \
	--gpu-memory-utilization 0.8 \
    --host 0.0.0.0   \
	--port 8000 \
    --load-format fastsafetensors   \
	--attention-backend flashinfer \
	--enable-prefix-caching \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --max-model-len 120000


