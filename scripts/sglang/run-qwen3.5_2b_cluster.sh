#!/bin/bash
# Multi-node SGLang: set NCCL / GLOO / MASTER in the Launch tab (cluster section) or export before running.
# On the head node use --node-rank 0 and your head NIC; on workers use --node-rank 1..N-1 and that host's NIC.
MODEL="Qwen/Qwen3.5-2B"
SERVED_MODEL_NAME="qwen3.5-2b"

python3 -m sglang.launch_server \
  --model-path ${MODEL} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --enable-metrics \
  --attention-backend triton 