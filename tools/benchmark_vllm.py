#!/usr/bin/env python3
"""Run ``vllm bench serve`` against a vLLM OpenAI-compatible server (default :8000).

Maps the same env knobs as the SGLang benchmark where applicable. Resolves the API model id from
``--model``, ``BENCHMARK_SERVED_MODEL``, or ``GET /v1/models``.

* ``--model`` / ``BENCHMARK_HF_MODEL`` (else ``--tokenizer``) — passed to ``vllm bench serve --model``
  for tokenizer / dataset setup when it differs from the served name.
* ``--served-model-name`` — API ``model`` field; defaults to the resolved served id.

Env (optional): BENCHMARK_BASE_URL (else VLLM_BASE_URL, else http://127.0.0.1:8000), BENCHMARK_BACKEND
(default ``openai-chat``), BENCHMARK_DATASET, BENCHMARK_NUM_PROMPTS, BENCHMARK_RANDOM_INPUT_LEN,
BENCHMARK_RANDOM_OUTPUT_LEN, BENCHMARK_SERVED_MODEL, BENCHMARK_HF_MODEL, BENCHMARK_TOKENIZER,
BENCHMARK_MAX_CONCURRENCY, BENCHMARK_EXTRA_REQUEST_BODY (merged into ``--extra-body``),
VLLM_BENCH_CMD (override executable, e.g. ``/opt/venv/bin/vllm``).

This script does **not** inject SGLang-specific ``separate_reasoning`` / Qwen ``enable_thinking`` defaults;
add them via ``BENCHMARK_EXTRA_REQUEST_BODY`` or ``--extra-body`` if your server needs them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

from benchmark_common import (
    env_int,
    env_optional_int,
    fetch_served_model_id,
    load_json_object,
    pop_json_flag_from_argv,
)

PROG = "benchmark_vllm.py"

DEFAULT_BASE = (
    os.environ.get("BENCHMARK_BASE_URL", "").strip()
    or os.environ.get("VLLM_BASE_URL", "").strip()
    or "http://127.0.0.1:8000"
)
DEFAULT_BACKEND = os.environ.get("BENCHMARK_BACKEND", "openai-chat")
DEFAULT_TOKENIZER = os.environ.get("BENCHMARK_TOKENIZER", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")

_DATASETS_WITH_RANDOM_LEN = frozenset({"random", "random-mm"})


def _vllm_cli_prefix() -> list[str]:
    raw = (os.environ.get("VLLM_BENCH_CMD") or "").strip()
    if raw:
        return shlex.split(raw)
    exe = shutil.which("vllm")
    if exe:
        return [exe]
    print(
        f"{PROG}: `vllm` not found on PATH. Install vLLM in this environment or set VLLM_BENCH_CMD.",
        file=sys.stderr,
    )
    raise SystemExit(127)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thin wrapper around `vllm bench serve` for the vLLM stack.",
        prog=PROG,
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"Server base URL (default {DEFAULT_BASE!r}; BENCHMARK_BASE_URL or VLLM_BASE_URL).",
    )
    p.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"bench backend (default {DEFAULT_BACKEND!r} or BENCHMARK_BACKEND).",
    )
    p.add_argument(
        "--dataset-name",
        default=os.environ.get("BENCHMARK_DATASET", "random"),
        help="Dataset name (default: random, or BENCHMARK_DATASET).",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=env_int("BENCHMARK_NUM_PROMPTS", 3),
        help="Prompt count (default 3 or BENCHMARK_NUM_PROMPTS).",
    )
    p.add_argument(
        "--random-input-len",
        type=int,
        default=env_int("BENCHMARK_RANDOM_INPUT_LEN", 128),
        help="For random / random-mm: input tokens (default 128).",
    )
    p.add_argument(
        "--random-output-len",
        type=int,
        default=env_int("BENCHMARK_RANDOM_OUTPUT_LEN", 128),
        help="For random / random-mm: output tokens (default 128).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=env_optional_int("BENCHMARK_MAX_CONCURRENCY"),
        help="Cap concurrent requests (optional; BENCHMARK_MAX_CONCURRENCY).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("BENCHMARK_SERVED_MODEL", "") or os.environ.get("BENCHMARK_MODEL", "")
        or "",
        metavar="SERVED_ID",
        help="Served model id for the API (BENCHMARK_SERVED_MODEL / BENCHMARK_MODEL); optional if /v1/models works.",
    )
    p.add_argument(
        "--hf-model",
        default=os.environ.get("BENCHMARK_HF_MODEL", "") or "",
        metavar="HF_REPO",
        help="HF repo for bench --model when it differs from the API id; default BENCHMARK_HF_MODEL or --tokenizer.",
    )
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer id or path for synthetic prompts (BENCHMARK_TOKENIZER).",
    )
    p.add_argument(
        "--extra-body",
        default=None,
        metavar="JSON",
        help="Merged into vLLM --extra-body after env (same object as BENCHMARK_EXTRA_REQUEST_BODY).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args, rest = parser.parse_known_args()

    rest_extra, rest = pop_json_flag_from_argv(rest, "--extra-body", PROG)

    tokenizer = args.tokenizer.strip() or DEFAULT_TOKENIZER
    hf_for_bench = (args.hf_model or "").strip() or tokenizer

    served = (args.model or "").strip() or os.environ.get("BENCHMARK_SERVED_MODEL", "").strip()
    if not served:
        served = fetch_served_model_id(args.base_url) or ""

    if not served:
        print(
            f"{PROG}: could not resolve served model id. Set --model, BENCHMARK_SERVED_MODEL, "
            "or ensure GET {}/v1/models returns a model.".format(args.base_url.rstrip("/")),
            file=sys.stderr,
        )
        raise SystemExit(2)

    cmd: list[str] = _vllm_cli_prefix() + [
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--dataset-name",
        args.dataset_name,
        "--model",
        hf_for_bench,
        "--served-model-name",
        served,
        "--tokenizer",
        tokenizer,
        "--num-prompts",
        str(args.num_prompts),
    ]

    if args.dataset_name in _DATASETS_WITH_RANDOM_LEN:
        cmd.extend(
            [
                "--random-input-len",
                str(args.random_input_len),
                "--random-output-len",
                str(args.random_output_len),
            ]
        )

    if args.max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(args.max_concurrency)])

    extra: dict = {}
    env_extra = (os.environ.get("BENCHMARK_EXTRA_REQUEST_BODY") or "").strip()
    if env_extra:
        extra.update(load_json_object("BENCHMARK_EXTRA_REQUEST_BODY", env_extra, PROG))
    if args.extra_body:
        extra.update(load_json_object("--extra-body", args.extra_body, PROG))
    extra.update(rest_extra)
    if extra:
        cmd.extend(["--extra-body", json.dumps(extra)])

    cmd.extend(rest)

    print("+ " + " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
