/**
 * Build a host-side `docker run` script for vLLM stack presets from repo `containers/vllm/*.sh`,
 * mirroring the Launch tab pattern (rendered script under `.monitor/`).
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type VllmStackPresetForRender = {
  /** API preset id (e.g. `vllm_tf5`); recorded in script header only. */
  id: string;
  /** Docker container name — used in the output filename so it matches `docker ps`. */
  containerName: string;
  matchesScript: string;
  image: string;
  extraEnv: readonly string[];
};

export type RenderVllmStackDockerInput = {
  preset: VllmStackPresetForRender;
  repoRoot: string;
  hostPublish: string;
  shm: string;
  clusterStackEnv: boolean;
  clusterRuntime: boolean;
  clusterDockerEnv: Record<string, string>;
  hfToken?: string;
};

function bashSingleQuoted(s: string): string {
  return `'${s.replace(/'/g, `'\"'\"'`)}'`;
}

/** True if `text` already passes `--network host` or `--network=host` to docker run. */
function scriptDeclaresDockerNetworkHost(text: string): boolean {
  // Do not use `\b` before `--`: space and `-` are both non-word chars, so `\b--network`
  // fails on indented lines like `    --network host \`.
  return /(?:^|\s)--network(?:\s+|=)host\b/.test(text);
}

function injectLinesBeforeImage(lines: string[], image: string, inject: string[]): string[] {
  const imageRe = new RegExp(
    `^\\s*${image.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*\\\\?\\s*$`,
  );
  let idx = lines.findIndex((l) => imageRe.test(l));
  if (idx < 0) {
    idx = lines.findIndex((l) => {
      const t = l.trim();
      return t === image || t.startsWith(`${image} `);
    });
  }
  if (idx < 0) return lines;
  return [...lines.slice(0, idx), ...inject, ...lines.slice(idx)];
}

function ensureClusterRuntimeFlags(
  lines: string[],
  opts: { hasIb: boolean },
): string[] {
  const text = lines.join("\n");
  if (!text.includes("--privileged")) {
    const gpus = lines.findIndex((l) => /docker\s+run\s+.*--gpus\s+all\b/.test(l));
    if (gpus >= 0) {
      const inject: string[] = [];
      if (!scriptDeclaresDockerNetworkHost(text)) {
        inject.push("    --network host \\");
      }
      inject.push("    --privileged \\");
      if (opts.hasIb) {
        inject.push("    -v /dev/infiniband:/dev/infiniband \\");
      }
      inject.push("    --ulimit memlock=-1:-1 \\");
      return [...lines.slice(0, gpus + 1), ...inject, ...lines.slice(gpus + 1)];
    }
  }
  return lines;
}

/** With cluster runtime + host network, omit `-p` (matches programmatic stack-run). */
function maybeDropPublishForCluster(lines: string[], clusterRuntime: boolean): string[] {
  if (!clusterRuntime) return lines;
  if (!lines.some((l) => scriptDeclaresDockerNetworkHost(l))) return lines;
  return lines.filter((l) => !/^\s*-p\s+/.test(l));
}

/** If both base script and cluster injection added `--network host`, keep the first only. */
function dedupeDockerNetworkHostLines(lines: string[]): string[] {
  let seen = false;
  return lines.filter((l) => {
    if (/^\s*--network(?:\s+|=)host\b/.test(l)) {
      if (seen) return false;
      seen = true;
    }
    return true;
  });
}

/** Use $VLLM_IMAGE in docker run so the host can override the image (e.g. export VLLM_IMAGE=... in ~/.bashrc). */
function replacePresetImageWithVllmEnv(lines: string[], image: string): string[] {
  const esc = image.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`^\\s*${esc}(\\s*\\\\)?\\s*$`);
  return lines.map((l) => {
    if (re.test(l)) {
      return l.replace(image, '"$VLLM_IMAGE"');
    }
    return l;
  });
}

export function writeVllmStackDockerScript(
  input: RenderVllmStackDockerInput,
): { ok: true; scriptPath: string } | { ok: false; error: string } {
  const basePath = path.join(input.repoRoot, input.preset.matchesScript);
  if (!fs.existsSync(basePath)) {
    return {
      ok: false,
      error: `Base container script missing: ${input.preset.matchesScript} (expected at ${basePath}).`,
    };
  }

  let body = fs.readFileSync(basePath, "utf8").replace(/\r\n/g, "\n").trimEnd();

  body = body.replace(/^docker run\b/m, "docker run -d");
  body = body.replace(/-v\s+\$\(pwd\):\/workspace/g, '-v "$REPO_ROOT:/workspace"');
  body = body.replace(
    /-v\s+~\/\.cache\/huggingface:\/root\/\.cache\/huggingface/g,
    '-v "$HF_CACHE:/root/.cache/huggingface"',
  );
  body = body.replace(/-p\s+8000:8000/g, '-p "${STACK_HOST_PORT}:8000"');
  body = body.replace(/--shm-size\s+\S+/g, '--shm-size "$STACK_SHM_SIZE"');
  body = body.replace(/([ \t]*)-it([ \t]+)/g, "$1");
  body = body.replace(/\bbash\s*$/m, "sleep infinity");

  let lines = body.split("\n");

  if (input.clusterRuntime) {
    lines = ensureClusterRuntimeFlags(lines, { hasIb: fs.existsSync("/dev/infiniband") });
    lines = maybeDropPublishForCluster(lines, true);
    lines = dedupeDockerNetworkHostLines(lines);
  }

  const injectEnv: string[] = [];
  if (input.clusterStackEnv) {
    for (const [k, v] of Object.entries(input.clusterDockerEnv)) {
      if (!v.trim()) continue;
      injectEnv.push(`    -e ${bashSingleQuoted(`${k}=${v}`)} \\`);
    }
  }
  for (const e of input.preset.extraEnv) {
    const t = e.trim();
    if (!t) continue;
    injectEnv.push(`    -e ${bashSingleQuoted(t)} \\`);
  }
  if (injectEnv.length > 0) {
    lines = injectLinesBeforeImage(lines, input.preset.image, injectEnv);
  }

  lines = replacePresetImageWithVllmEnv(lines, input.preset.image);

  const monitorDir = path.join(input.repoRoot, ".monitor");
  fs.mkdirSync(monitorDir, { recursive: true });
  const scriptPath = path.join(
    monitorDir,
    `monitor-stack-${input.preset.containerName}.rendered.sh`,
  );

  const header: string[] = [
    "#!/usr/bin/env bash",
    "# Generated by spark-stack-monitor — source: " + input.preset.matchesScript,
    `# Preset id: ${input.preset.id}  container: ${input.preset.containerName}`,
    "set -euo pipefail",
    `REPO_ROOT=${bashSingleQuoted(input.repoRoot)}`,
    `HF_CACHE=${bashSingleQuoted(path.join(os.homedir(), ".cache", "huggingface"))}`,
    `STACK_HOST_PORT=${bashSingleQuoted(input.hostPublish)}`,
    `STACK_SHM_SIZE=${bashSingleQuoted(input.shm)}`,
  ];
  if (input.hfToken) {
    header.push(`export HF_TOKEN=${bashSingleQuoted(input.hfToken)}`);
  }
  header.push(`VLLM_IMAGE="\${VLLM_IMAGE:-${input.preset.image}}"`);
  header.push("export VLLM_IMAGE");
  header.push("", ...lines);
  header.push("");

  fs.writeFileSync(scriptPath, header.join("\n"), { mode: 0o644 });
  fs.chmodSync(scriptPath, 0o755);

  return { ok: true, scriptPath };
}
