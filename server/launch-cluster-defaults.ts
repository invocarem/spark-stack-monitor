/**
 * Defaults for the Launch tab “Multi-node cluster” block, from `process.env`
 * (typically set in repo-root `.env` — see `.env.example`).
 */

function pick(
  monitorKey: string | undefined,
  fallbackKey: string | undefined,
): string {
  const a = monitorKey?.trim() ?? "";
  const b = fallbackKey?.trim() ?? "";
  return a.length > 0 ? a : b;
}

export type LaunchClusterDefaultsResponse = {
  /** `export` keys sent with `POST /api/launch` when the user runs a script */
  launchEnv: Record<string, string>;
  distInit: string;
  nnodes: string;
  nodeRank: string;
  /** When true, the UI checks “Apply cluster options” on load */
  applyCluster: boolean;
};

export function getLaunchClusterDefaultsFromEnv(): LaunchClusterDefaultsResponse {
  const nccl = pick(
    process.env.MONITOR_CLUSTER_NCCL_SOCKET_IFNAME,
    process.env.NCCL_SOCKET_IFNAME,
  );
  const gloo = pick(
    process.env.MONITOR_CLUSTER_GLOO_SOCKET_IFNAME,
    process.env.GLOO_SOCKET_IFNAME,
  );
  const masterAddr = pick(
    process.env.MONITOR_CLUSTER_MASTER_ADDR,
    process.env.MASTER_ADDR,
  );
  const masterPort = pick(
    process.env.MONITOR_CLUSTER_MASTER_PORT,
    process.env.MASTER_PORT,
  );

  const launchEnv: Record<string, string> = {};
  if (nccl) launchEnv.NCCL_SOCKET_IFNAME = nccl;
  if (gloo) launchEnv.GLOO_SOCKET_IFNAME = gloo;
  if (masterAddr) launchEnv.MASTER_ADDR = masterAddr;
  if (masterPort) launchEnv.MASTER_PORT = masterPort;

  const distInit =
    process.env.MONITOR_CLUSTER_DIST_INIT_ADDR?.trim() ||
    process.env.SGLANG_DIST_INIT_ADDR?.trim() ||
    "";
  const nnodes = process.env.MONITOR_CLUSTER_NNODES?.trim() || "";
  const nodeRank = process.env.MONITOR_CLUSTER_NODE_RANK?.trim() || "";

  const applyFlag = process.env.MONITOR_CLUSTER_APPLY?.trim().toLowerCase();
  const hasAny =
    Object.keys(launchEnv).length > 0 || Boolean(distInit || nnodes || nodeRank);
  let applyCluster: boolean;
  if (applyFlag === "0" || applyFlag === "false" || applyFlag === "no") {
    applyCluster = false;
  } else if (applyFlag === "1" || applyFlag === "true" || applyFlag === "yes") {
    applyCluster = true;
  } else {
    applyCluster = hasAny;
  }

  return {
    launchEnv,
    distInit,
    nnodes,
    nodeRank,
    applyCluster,
  };
}
