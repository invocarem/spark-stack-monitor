#!/bin/bash
#
# Run the vLLM Docker image with GPU + HF cache, without starting Ray.
# Use this when --distributed-executor-backend mp is enough (typical: single-node
# multi-GPU, or multi-node mp with --nnodes / --node-rank on each host).
#
# After the container is up, run vllm serve inside it (e.g. via docker exec),
# with --distributed-executor-backend mp instead of ray.
#
# Usage:
#   bash run_vllm_container.sh docker_image /abs/path/to/huggingface/cache \
#        [--container-name|-n NAME] [extra docker run args, e.g. -e VAR=val ...]
#
# Example (same cache and env style as head.sh):
#   bash run_vllm_container.sh $VLLM_IMAGE ~/.cache/huggingface \
#     -n vllm_node -e VLLM_HOST_IP=192.168.1.1 -e NCCL_SOCKET_IFNAME=eth0

if [ $# -lt 2 ]; then
    echo "Usage: $0 docker_image path_to_hf_home [--container-name|-n NAME] [additional docker args...]"
    exit 1
fi

DOCKER_IMAGE="$1"
PATH_TO_HF_HOME="$2"
shift 2

ADDITIONAL_ARGS=("$@")

CONTAINER_NAME=""
NEW_ARGS=()
i=0
while [ "${i}" -lt "${#ADDITIONAL_ARGS[@]}" ]; do
    arg="${ADDITIONAL_ARGS[$i]}"
    case "${arg}" in
        --container-name)
            i=$((i + 1))
            if [ "${i}" -ge "${#ADDITIONAL_ARGS[@]}" ]; then
                echo "Error: --container-name requires a value"
                exit 1
            fi
            CONTAINER_NAME="${ADDITIONAL_ARGS[$i]}"
            i=$((i + 1))
            ;;
        -n)
            i=$((i + 1))
            if [ "${i}" -ge "${#ADDITIONAL_ARGS[@]}" ]; then
                echo "Error: -n requires a container name"
                exit 1
            fi
            CONTAINER_NAME="${ADDITIONAL_ARGS[$i]}"
            i=$((i + 1))
            ;;
        *)
            NEW_ARGS+=("${arg}")
            i=$((i + 1))
            ;;
    esac
done
ADDITIONAL_ARGS=("${NEW_ARGS[@]}")

if [ -z "${CONTAINER_NAME}" ]; then
    CONTAINER_NAME="vllm-no-ray-${RANDOM}"
fi

# Default repo mount for /workspace (same intent as run_cluster.sh).
: "${MONITOR_REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cleanup() {
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

if [ -t 1 ]; then
  DOCKER_TTY_FLAGS=(-it)
else
  DOCKER_TTY_FLAGS=(-i)
fi

docker run "${DOCKER_TTY_FLAGS[@]}" --rm \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    -v "${MONITOR_REPO_ROOT}:/workspace" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "sleep infinity"
