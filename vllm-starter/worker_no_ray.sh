# Partner to head_no_ray.sh on the second machine: same idea, no Ray.
# For multi-node tensor parallel with mp, set MASTER_ADDR / MASTER_PORT and
# launch vllm on both nodes with --nnodes / --node-rank (see qwen3.5_397b_autoround_mp.sh).

export MN_IF_NAME=enp1s0f0np0
export VLLM_HOST_IP=192.168.100.12

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP (no Ray)"

bash "$(dirname "$0")/run_vllm_container.sh" "$VLLM_IMAGE" ~/.cache/huggingface \
  -n vllm_node_tf5_worker \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_SOCKET_IFNAME=^lo,docker0 \
  -e NCCL_DEBUG=INFO
