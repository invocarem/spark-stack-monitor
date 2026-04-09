docker run --gpus all \
    --name vllm_node_tf5 \
    --network host \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    -it --rm \
    vllm-node-tf5:latest \
    bash
