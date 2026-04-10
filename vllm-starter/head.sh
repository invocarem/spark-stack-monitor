# Use the interface that has 192.168.100.11
export MN_IF_NAME=enp1s0f1np1  # This is correct for spark1 on 100.xx network
export VLLM_HOST_IP=192.168.100.11

# Stop existing containers
#docker stop $(docker ps -aq) 2>/dev/null || true
#docker rm $(docker ps -aq) 2>/dev/null || true
echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

# Restart head node
bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP \
  --head ~/.cache/huggingface \
  -n vllm_node_tf5 \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_SOCKET_IFNAME=^lo,docker0 \
  -e NCCL_DEBUG=INFO


