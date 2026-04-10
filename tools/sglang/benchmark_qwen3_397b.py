#!/usr/bin/env python3
"""Run a fixed SGLang bench_serving profile for Qwen3.5-397B GPTQ.

Equivalent command:
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://100.109.56.33:30000 \
  --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --served-model-name qwen3.5-397b \
  --num-prompts 10 \
  --max-concurrency 3 \
  --random-input-len 512 \
  --random-output-len 256 \
  --extra-request-body '{"separate_reasoning": false, "chat_template_kwargs": {"enable_thinking": false}}'
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def main() -> int:
    base_url = os.environ.get("QWEN397_BENCH_BASE_URL", "http://100.109.56.33:30000").strip()
    served_model = os.environ.get("QWEN397_BENCH_SERVED_MODEL", "qwen3.5-397b").strip()
    hf_model = os.environ.get("QWEN397_BENCH_HF_MODEL", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4").strip()

    extra_body = {
        "separate_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        base_url,
        "--model",
        hf_model,
        "--served-model-name",
        served_model,
        "--num-prompts",
        "10",
        "--max-concurrency",
        "3",
        "--dataset-name",
        "random",
        "--random-input-len",
        "512",
        "--random-output-len",
        "256",
        "--extra-request-body",
        json.dumps(extra_body),
    ]

    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
