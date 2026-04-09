/**
 * vLLM-specific launch rendering: cluster flags for `vllm serve` (see vLLM CLI ParallelConfig).
 */

import type { LaunchArgPair } from "../launch-types.js";

/**
 * Order for injecting missing cluster args into a rendered `vllm serve \\` block.
 * `--dist-init-addr` is SGLang-only and is stripped by {@link filterVllmLaunchArgOverrides}.
 */
export const VLLM_CLUSTER_INJECT_ARG_KEY_ORDER = [
  "--distributed-executor-backend",
  "--master-addr",
  "--master-port",
  "--nnodes",
  "--node-rank",
] as const;

export function vllmClusterArgSortIndex(key: string): number {
  const idx = (VLLM_CLUSTER_INJECT_ARG_KEY_ORDER as readonly string[]).indexOf(key);
  return idx === -1 ? Number.POSITIVE_INFINITY : idx;
}

/** Drop SGLang-only flags before rendering `vllm serve`. */
export function filterVllmLaunchArgOverrides(
  argOverrides: LaunchArgPair[] | undefined,
): LaunchArgPair[] | undefined {
  if (!argOverrides?.length) return argOverrides;
  return argOverrides.filter((a) => a.key !== "--dist-init-addr");
}
