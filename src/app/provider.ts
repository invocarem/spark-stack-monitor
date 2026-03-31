export type MonitorProvider = "sglang" | "vllm";

const STORAGE_KEY = "monitor-provider";
const CHANGE_EVENT = "monitor-provider-changed";
const DEFAULT_PROVIDER: MonitorProvider = "sglang";

const selProvider = document.querySelector<HTMLSelectElement>("#sel-provider");
const statusProvider = document.querySelector<HTMLSpanElement>("#status-provider");

function normalizeProvider(input: string | null | undefined): MonitorProvider {
  return input === "vllm" ? "vllm" : "sglang";
}

export function getMonitorProvider(): MonitorProvider {
  try {
    return normalizeProvider(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return DEFAULT_PROVIDER;
  }
}

export function setMonitorProvider(next: MonitorProvider): void {
  const value = normalizeProvider(next);
  try {
    window.localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // Ignore storage failures (e.g. privacy mode).
  }
  window.dispatchEvent(new CustomEvent<MonitorProvider>(CHANGE_EVENT, { detail: value }));
}

export function onMonitorProviderChange(listener: (provider: MonitorProvider) => void): void {
  window.addEventListener(CHANGE_EVENT, (ev: Event) => {
    const custom = ev as CustomEvent<MonitorProvider>;
    listener(normalizeProvider(custom.detail));
  });
}

function updateFooter(provider: MonitorProvider): void {
  if (selProvider) selProvider.value = provider;
  if (!statusProvider) return;
  if (provider === "vllm") {
    statusProvider.textContent = "vLLM selected. Metrics/config supported; chat and benchmark may be limited.";
    return;
  }
  statusProvider.textContent = "SGLang selected.";
}

export function withProviderQuery(path: string): string {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("provider", getMonitorProvider());
  return `${url.pathname}${url.search}`;
}

export function withProviderHeaders(
  headers: Record<string, string> = {},
): Record<string, string> {
  return {
    ...headers,
    "x-monitor-provider": getMonitorProvider(),
  };
}

export function initProviderFooter(): void {
  const current = getMonitorProvider();
  updateFooter(current);
  selProvider?.addEventListener("change", () => {
    const value = normalizeProvider(selProvider.value);
    setMonitorProvider(value);
  });
  onMonitorProviderChange((provider) => {
    updateFooter(provider);
  });
}
