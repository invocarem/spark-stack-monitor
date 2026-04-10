# Step 1: Get your container name
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')

# Step 2: Apply the patch inside the container
docker exec -it $VLLM_CONTAINER /bin/bash -c "
  FILE='/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py'
  if [ -f \"\$FILE\" ]; then
    sed -i 's/ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {/ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | {/g' \"\$FILE\"
    echo '✓ Patch applied successfully'
  else
    echo '✗ File not found'
  fi
"
