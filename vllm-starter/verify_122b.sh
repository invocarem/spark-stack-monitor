# From spark1 or any machine that can reach spark1:8000
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
    "prompt": "What is distributed computing?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
