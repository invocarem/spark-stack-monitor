# On spark1 (head node)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "container: $VLLM_CONTAINER"
#docker exec $VLLM_CONTAINER ping -c 2 192.168.100.12
docker exec $VLLM_CONTAINER curl -v --connect-timeout 5 http://192.168.100.12:8265
