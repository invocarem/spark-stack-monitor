# Same networking and NCCL env as head.sh, but no Ray — container only runs
# sleep infinity. Use qwen3.5_397b_autoround_mp.sh (docker exec) to start vLLM
# with --distributed-executor-backend mp.

export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=192.168.100.11

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP (no Ray)"

bash "$(dirname "$0")/run_vllm_container.sh" "$VLLM_IMAGE" ~/.cache/huggingface \
  -n vllm_node \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_SOCKET_IFNAME=^lo,docker0 \
  -e NCCL_DEBUG=INFO
