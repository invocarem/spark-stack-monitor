docker run --gpus all \
    --name sglang_node_tf5 \
    --network host \
    --shm-size 32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "NCCL_SOCKET_IFNAME=eth0" \
    --env "NCCL_DEBUG=INFO" \
    --env "NCCL_IB_DISABLE=1" \
    --env "MASTER_ADDR=192.168.100.11" \
    --env "MASTER_PORT=50000" \
    --env "WORLD_SIZE=2" \
    --ipc=host \
    -it --rm \
    scitrera/dgx-spark-sglang:0.5.9-t5 \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-35B-A3B \
        --served-model-name qwen3.5-35b \
        --tp-size 2 \
        --nnodes 2 \
        --node-rank 0 \
        --dist-init-addr 192.168.100.11:50000 \
        --host 0.0.0.0 \
        --port 30000 \
        --trust-remote-code