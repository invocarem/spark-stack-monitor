# Use the interface that has 192.168.100.12
export MN_IF_NAME=enp1s0f0np0  
export VLLM_HOST_IP=192.168.100.12
export HEAD_NODE_IP=192.168.100.11

# Stop existing containers
#docker stop $(docker ps -aq) 2>/dev/null || true
#docker rm $(docker ps -aq) 2>/dev/null || true

# Restart worker node
bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
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
