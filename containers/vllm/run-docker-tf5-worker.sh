# Ray worker — same flags as vllm-starter/worker.sh calling vllm-starter/run_cluster.sh (see that folder).
# Image defaults to vllm-node-tf5:latest; override with VLLM_IMAGE (e.g. from ~/.bashrc).
HEAD_NODE_IP="${HEAD_NODE_IP:-192.168.100.11}"
VLLM_HOST_IP="${VLLM_HOST_IP:-192.168.100.12}"
MN_IF_NAME="${MN_IF_NAME:-enp1s0f0np0}"

docker run --gpus all \
    --entrypoint /bin/bash \
    --network host \
    --name vllm_node_tf5_worker \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    --ipc=host \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    -e "VLLM_HOST_IP=${VLLM_HOST_IP}" \
    -e "UCX_NET_DEVICES=${MN_IF_NAME}" \
    -e "NCCL_SOCKET_IFNAME=${MN_IF_NAME}" \
    -e "GLOO_SOCKET_IFNAME=${MN_IF_NAME}" \
    -e "TP_SOCKET_IFNAME=${MN_IF_NAME}" \
    -e "NCCL_IB_DISABLE=0" \
    -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0" \
    -e "NCCL_SOCKET_IFNAME=^lo,docker0" \
    -e "NCCL_DEBUG=INFO" \
    -it \
    vllm-node-tf5:latest \
    -c "ray start --block --address=${HEAD_NODE_IP}:6379 --node-ip-address=${VLLM_HOST_IP}"
