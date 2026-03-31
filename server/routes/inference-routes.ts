import type { Context, Hono } from "hono";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import {
  fetchSglangMetrics,
  forwardChatCompletions,
  getSglangBaseUrl,
  getSglangMetricsUrl,
  runSglangBenchmark,
} from "../sglang.js";
import {
  fetchVllmMetrics,
  forwardVllmChatCompletions,
  getVllmBaseUrl,
  getVllmMetricsUrl,
  runVllmBenchmark,
} from "../vllm.js";

type ProviderId = "sglang" | "vllm";

function pickProvider(c: Context): ProviderId {
  const q = c.req.query("provider")?.trim().toLowerCase();
  if (q === "sglang" || q === "vllm") return q;
  const h = c.req.header("x-monitor-provider")?.trim().toLowerCase();
  if (h === "sglang" || h === "vllm") return h;
  const env = process.env.MONITOR_PROVIDER?.trim().toLowerCase();
  if (env === "sglang" || env === "vllm") return env;
  return "sglang";
}

function buildConfig(provider: ProviderId) {
  const metricsUrl = provider === "vllm" ? getVllmMetricsUrl() : getSglangMetricsUrl();
  const inferenceBaseUrl = provider === "vllm" ? getVllmBaseUrl() : getSglangBaseUrl();
  const u = new URL(metricsUrl);
  const defaultModel = process.env.SGLANG_DEFAULT_MODEL?.trim() || undefined;
  return {
    provider,
    metricsUrl,
    inferenceBaseUrl,
    host: u.host,
    hint:
      provider === "vllm"
        ? "vLLM metrics/config selected. Chat and benchmark are not yet wired in this unified monitor API."
        : "Launch SGLang with --enable-metrics (scripts in this repo include it). Prometheus text is served at /metrics on the server port.",
    ...(defaultModel ? { defaultModel } : {}),
  };
}

async function handleChatCompletions(c: Context, provider: ProviderId) {
  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400);
  }
  const result =
    provider === "vllm"
      ? await forwardVllmChatCompletions(body)
      : await forwardChatCompletions(body);
  if (!result.ok) {
    const status = (result.status ?? 502) as ContentfulStatusCode;
    const preview = result.bodyPreview;
    if (preview !== undefined) {
      try {
        return c.json(
          { error: result.error, detail: JSON.parse(preview) as unknown },
          status,
        );
      } catch {
        return c.json({ error: result.error, detail: preview }, status);
      }
    }
    return c.json({ error: result.error }, status);
  }
  return c.json(result.body, result.status as ContentfulStatusCode);
}

async function handleBenchmark(c: Context, provider: ProviderId) {
  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400);
  }
  const result = provider === "vllm" ? await runVllmBenchmark(body) : await runSglangBenchmark(body);
  if (!result.ok) {
    const status = result.status as ContentfulStatusCode;
    return c.json({ error: result.error }, status);
  }
  return c.json(result);
}

async function handleMetrics(c: Context, provider: ProviderId) {
  const result = provider === "vllm" ? await fetchVllmMetrics() : await fetchSglangMetrics();
  if (!result.ok) {
    return c.json(result, 502);
  }
  return c.json({ ...result, provider });
}

export function registerInferenceRoutes(app: Hono): void {
  // Core routes.
  app.get("/api/config", (c) => {
    const provider = pickProvider(c);
    try {
      return c.json(buildConfig(provider));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      return c.json({ error: message }, 400);
    }
  });
  app.get("/api/metrics", (c) => handleMetrics(c, pickProvider(c)));
  app.post("/api/chat/completions", (c) => handleChatCompletions(c, pickProvider(c)));
  app.post("/api/benchmark", (c) => handleBenchmark(c, pickProvider(c)));

  // Legacy aliases kept for compatibility while the frontend migrates.
  app.get("/api/sglang/config", (c) => {
    try {
      return c.json(buildConfig("sglang"));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      return c.json({ error: message }, 400);
    }
  });
  app.get("/api/sglang/metrics", (c) => handleMetrics(c, "sglang"));
  app.post("/api/sglang/chat/completions", (c) => handleChatCompletions(c, "sglang"));
  app.post("/api/sglang/benchmark", (c) => handleBenchmark(c, "sglang"));
}
