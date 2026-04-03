/**
 * Whitelisted `docker run` presets for the stack dev container (host API).
 * Each preset mirrors the flags in repo `containers/sglang/run-docker.sh` and
 * `containers/sglang/run-docker-openai.sh` (GPU, shm, port, mounts, env, image, name).
 * The monitor uses `docker run -d` with `sleep infinity` instead of `-it … bash` so
 * the process works without a TTY and the Launch tab can `docker exec`.
 *
 * vLLM testing lives on a separate page; see `vllm-stack.ts`.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { assertSafeContainerName, dockerHost } from "./docker.js";
import {
  getSglangStackDockerEnvForClusterRun,
  shouldInjectSglangStackClusterDockerEnv,
} from "./launch-cluster-defaults.js";
import { findRepoRoot } from "./repo-root.js";

export type StackProvider = "sglang" | "vllm";

export type StackPreset = {
  id: string;
  label: string;
  provider: StackProvider;
  /** For documentation only; same layout as `containers/*.sh`. */
  matchesScript: string;
  containerName: string;
  image: string;
  /** Extra `docker run -e` pairs after HF_TOKEN (e.g. OpenAI/tiktoken image). */
  extraEnv: readonly string[];
};

export const STACK_PRESETS: readonly StackPreset[] = [
  {
    id: "dgx_spark_tf5",
    label: "SciTrera DGX Spark SGLang (tf5)",
    provider: "sglang",
    matchesScript: "containers/sglang/run-docker.sh",
    containerName: "sglang_node_tf5",
    image: "scitrera/dgx-spark-sglang:0.5.9-t5",
    extraEnv: [],
  },
   {
    id: "dgx_spark_tf5_10",
    label: "SciTrera DGX Spark SGLang (tf5 10)",
    provider: "sglang",
    matchesScript: "containers/sglang/run-docker.sh",
    containerName: "sglang_node_tf5_10",
    image: "scitrera/dgx-spark-sglang:0.5.10rc0",
    extraEnv: ["SGLANG_ENABLE_SPEC_V2=1"],
    
  },
  {
    id: "lmsys_spark",
    label: "LM.Sys SGLang (spark)",
    provider: "sglang",
    matchesScript: "containers/sglang/run-docker-openai.sh",
    containerName: "sglang_node",
    image: "lmsysorg/sglang:spark",
    extraEnv: ["TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings"],
  },
  {
    id: "vllm_tf5",
    label: "vLLM Node (tf5)",
    provider: "vllm",
    matchesScript: "containers/vllm/run-docker-tf5.sh",
    containerName: "vllm_node_tf5",
    image: "vllm-node-tf5:latest",
    extraEnv: [],
  },
  {
    id: "vllm_node",
    label: "vLLM Node",
    provider: "vllm",
    matchesScript: "containers/vllm/run-docker.sh",
    containerName: "vllm_node",
    image: "vllm-node:latest",
    extraEnv: [],
  },
] as const;

const PRESET_BY_ID = new Map(STACK_PRESETS.map((p) => [p.id, p]));

const ALLOWED_NAMES = new Set(STACK_PRESETS.map((p) => p.containerName));

/** Published host port for SGLang stack presets (maps to container :30000; matches `scripts/sglang/*.sh`). */
function sglangStackHostPort(): string {
  const n = Number(process.env.MONITOR_STACK_HOST_PORT ?? "30000");
  if (!Number.isFinite(n) || n < 1 || n > 65535) return "30000";
  return String(Math.trunc(n));
}

/** Published host port for vLLM stack presets (maps to container :8000). */
function vllmStackHostPort(): string {
  const n = Number(process.env.MONITOR_STACK_HOST_PORT ?? "8000");
  if (!Number.isFinite(n) || n < 1 || n > 65535) return "8000";
  return String(Math.trunc(n));
}

function shmSize(): string {
  const s = process.env.MONITOR_STACK_SHM_SIZE?.trim();
  return s || "32g";
}

/** Safe single-quoted string for generated bash scripts. */
function shellSingleQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export type StackDockerArgvParts = {
  /** Full argv for `docker …` (host), including HF_TOKEN when set in the monitor process env. */
  dockerArgv: string[];
  /** Through `--rm` (order matches `docker run`). */
  scriptCore: string[];
  /** Cluster / preset `-e` pairs after HF_TOKEN in `docker run`. */
  scriptMid: string[];
  /** `image sleep infinity`. */
  scriptTail: string[];
  /** When true, generated script inserts HF after `scriptCore` via `if [ -n "${HF_TOKEN:-}" ]; then …`. */
  scriptUsesHfTokenConditional: boolean;
};

/**
 * Builds `docker run …` argv for stack presets. Separates HF_TOKEN so we can emit a host `.sh` file
 * without writing secrets to disk (same idea as Launch writing under `/workspace/.monitor/`).
 */
export function buildStackDockerRunArgvParts(
  preset: StackPreset,
  repoRoot: string,
): StackDockerArgvParts {
  const hfCache = path.join(os.homedir(), ".cache", "huggingface");
  const shm = shmSize();
  const hostPublish =
    preset.provider === "vllm" ? vllmStackHostPort() : sglangStackHostPort();
  const containerPublish = preset.provider === "vllm" ? "8000" : "30000";
  const clusterStackEnv =
    preset.provider === "sglang" && shouldInjectSglangStackClusterDockerEnv();

  const scriptCore: string[] = ["run", "-d", "--gpus", "all"];
  if (clusterStackEnv) {
    scriptCore.push("--network", "host");
  }
  scriptCore.push(
    "--name",
    preset.containerName,
    "--shm-size",
    shm,
    "-p",
    `${hostPublish}:${containerPublish}`,
    "-v",
    `${hfCache}:/root/.cache/huggingface`,
    "-v",
    `${repoRoot}:/workspace`,
    "--ipc=host",
    "--rm",
  );

  const scriptMid: string[] = [];
  if (clusterStackEnv) {
    for (const [k, v] of Object.entries(getSglangStackDockerEnvForClusterRun())) {
      scriptMid.push("-e", `${k}=${v}`);
    }
  }
  for (const e of preset.extraEnv) {
    scriptMid.push("-e", e);
  }
  const scriptTail: string[] = [preset.image, "sleep", "infinity"];

  const token = process.env.HF_TOKEN?.trim();
  const scriptUsesHfTokenConditional = Boolean(token);

  const mid: string[] = [];
  if (token) {
    mid.push("-e", `HF_TOKEN=${token}`);
  }
  mid.push(...scriptMid);

  const dockerArgv = [...scriptCore, ...mid, ...scriptTail];

  return {
    dockerArgv,
    scriptCore,
    scriptMid,
    scriptTail,
    scriptUsesHfTokenConditional,
  };
}

function bashDockerArgsBlock(lines: string[]): string {
  return ["docker_args=(", ...lines.map((a) => `  ${shellSingleQuote(a)}`), ")"].join("\n");
}

function writeStackDockerRunMonitorScript(
  repoRoot: string,
  preset: StackPreset,
  parts: StackDockerArgvParts,
): string {
  const dir = path.join(repoRoot, ".monitor");
  fs.mkdirSync(dir, { recursive: true });
  const rel = `.monitor/monitor-stack-${preset.id}.sh`;
  const filePath = path.join(repoRoot, rel);

  const header = [
    "#!/usr/bin/env bash",
    "# Generated by spark-stack-monitor (Container tab → Start).",
    `# Preset: ${preset.id} — ${preset.label}`,
    `# Mirrors ${preset.matchesScript}; same flags as the API (sleep infinity for idle stack).`,
    "set -euo pipefail",
    "",
  ].join("\n");

  let body: string;
  if (parts.scriptUsesHfTokenConditional) {
    const coreBlock = bashDockerArgsBlock(parts.scriptCore);
    const restLines = [...parts.scriptMid, ...parts.scriptTail].map((a) => `  ${shellSingleQuote(a)}`);
    body = [
      "# HF_TOKEN is inserted after --rm only if set in the environment (not stored in this file).",
      coreBlock,
      'if [ -n "${HF_TOKEN:-}" ]; then',
      '  docker_args+=(-e "HF_TOKEN=$HF_TOKEN")',
      "fi",
      "docker_args+=(",
      ...restLines,
      ")",
      'exec docker "${docker_args[@]}"',
      "",
    ].join("\n");
  } else {
    const flat = [...parts.scriptCore, ...parts.scriptMid, ...parts.scriptTail];
    const quoted = flat.map(shellSingleQuote).join(" ");
    body = `exec docker ${quoted}\n`;
  }

  fs.writeFileSync(filePath, `${header}${body}`, { mode: 0o755 });
  return rel;
}

function writeStackResumeMonitorScript(repoRoot: string, preset: StackPreset): string {
  const dir = path.join(repoRoot, ".monitor");
  fs.mkdirSync(dir, { recursive: true });
  const rel = `.monitor/monitor-stack-${preset.id}-resume.sh`;
  const filePath = path.join(repoRoot, rel);
  const content = [
    "#!/usr/bin/env bash",
    "# Generated by spark-stack-monitor — resumes a stopped stack container.",
    `# Preset: ${preset.id} (${preset.containerName})`,
    "set -euo pipefail",
    `# Full create recipe: .monitor/monitor-stack-${preset.id}.sh`,
    `exec docker ${shellSingleQuote("start")} ${shellSingleQuote(preset.containerName)}`,
    "",
  ].join("\n");
  fs.writeFileSync(filePath, content, { mode: 0o755 });
  return rel;
}

async function containerState(
  name: string,
): Promise<{ kind: "missing" } | { kind: "running" } | { kind: "stopped" }> {
  const r = await dockerHost(["inspect", "-f", "{{.State.Running}}", name]);
  if (r.code !== 0) return { kind: "missing" };
  return r.stdout.trim() === "true" ? { kind: "running" } : { kind: "stopped" };
}

export function getStackPreset(id: string): StackPreset | undefined {
  return PRESET_BY_ID.get(id);
}

export function listStackPresets(provider: StackProvider): StackPreset[] {
  return STACK_PRESETS.filter((p) => p.provider === provider);
}

export type RunStackResult =
  | {
      ok: true;
      container: string;
      started: boolean;
      message: string;
      /** Repo-relative path to a generated helper script under `.monitor/`, when written. */
      scriptRelPath?: string;
    }
  | { ok: false; error: string; stderr?: string };

export async function runStackPreset(presetId: string): Promise<RunStackResult> {
  const preset = getStackPreset(presetId);
  if (!preset) {
    return { ok: false, error: "Unknown stack preset." };
  }
  try {
    assertSafeContainerName(preset.containerName);
  } catch {
    return { ok: false, error: "Invalid container name in preset." };
  }

  const state = await containerState(preset.containerName);
  if (state.kind === "running") {
    return {
      ok: true,
      container: preset.containerName,
      started: false,
      message: `Container ${preset.containerName} is already running.`,
    };
  }

  const repoRoot = findRepoRoot();

  if (state.kind === "stopped") {
    let resumeRel: string;
    try {
      resumeRel = writeStackResumeMonitorScript(repoRoot, preset);
    } catch (e) {
      return {
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      };
    }
    const start = await dockerHost(["start", preset.containerName]);
    if (start.code !== 0) {
      const err = (start.stderr.trim() || start.stdout.trim()).slice(0, 800);
      return {
        ok: false,
        error: err || `docker start failed (exit ${start.code ?? "?"})`,
        stderr: start.stderr.trim() || undefined,
      };
    }
    return {
      ok: true,
      container: preset.containerName,
      started: true,
      scriptRelPath: resumeRel,
      message: `Wrote ${resumeRel}; started existing container ${preset.containerName}.`,
    };
  }

  const parts = buildStackDockerRunArgvParts(preset, repoRoot);
  let dockerRunRel: string;
  try {
    dockerRunRel = writeStackDockerRunMonitorScript(repoRoot, preset, parts);
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }

  const hostPublish =
    preset.provider === "vllm" ? vllmStackHostPort() : sglangStackHostPort();
  const containerPublish = preset.provider === "vllm" ? "8000" : "30000";
  const clusterStackEnv =
    preset.provider === "sglang" && shouldInjectSglangStackClusterDockerEnv();

  const run = await dockerHost(parts.dockerArgv);
  if (run.code !== 0) {
    const err = (run.stderr.trim() || run.stdout.trim()).slice(0, 1200);
    return {
      ok: false,
      error: err || `docker run failed (exit ${run.code ?? "?"})`,
      stderr: run.stderr.trim() || undefined,
    };
  }

  return {
    ok: true,
    container: preset.containerName,
    started: true,
    scriptRelPath: dockerRunRel,
    message: `Wrote ${dockerRunRel}; created and started ${preset.containerName} (same flags as ${preset.matchesScript}; monitor uses sleep infinity).${clusterStackEnv ? " Cluster `.env` NCCL/distributed env and --network host applied." : ""} Host port ${hostPublish}→${containerPublish}; repo at /workspace.`,
  };
}

export type StopStackResult =
  | { ok: true; message: string }
  | { ok: false; error: string; stderr?: string };

/** Stop a stack container by name (whitelist only — names from STACK_PRESETS). */
export async function stopStackContainer(containerName: string): Promise<StopStackResult> {
  const name = containerName.trim();
  if (!ALLOWED_NAMES.has(name)) {
    return { ok: false, error: "Container name is not a known stack preset." };
  }
  try {
    assertSafeContainerName(name);
  } catch {
    return { ok: false, error: "Invalid container name." };
  }

  const state = await containerState(name);
  if (state.kind === "missing") {
    return { ok: true, message: `No container named ${name}.` };
  }
  if (state.kind === "stopped") {
    return { ok: true, message: `Container ${name} is already stopped.` };
  }

  const stop = await dockerHost(["stop", name]);
  if (stop.code !== 0) {
    const err = (stop.stderr.trim() || stop.stdout.trim()).slice(0, 800);
    return {
      ok: false,
      error: err || `docker stop failed (exit ${stop.code ?? "?"})`,
      stderr: stop.stderr.trim() || undefined,
    };
  }
  return { ok: true, message: `Stopped ${name} (with --rm it may be removed).` };
}
