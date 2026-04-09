 docker run --privileged --gpus all -it --rm \
  --ipc=host \
  --network host \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node ./run-cluster-node.sh \
    --role head \
    --host-ip 192.168.100.11 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1,roceP2p1s0f1  
