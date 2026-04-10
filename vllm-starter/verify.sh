# From spark1 or any machine that can reach spark1:8000
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-2B",
    "prompt": "What is distributed computing?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
