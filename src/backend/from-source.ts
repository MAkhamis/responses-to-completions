import type { BackendAdapter } from "./adapter.js";
import { OllamaAdapter, type OllamaAdapterOptions } from "./ollama.js";
import {
  OpenAICompatAdapter,
  type OpenAICompatAdapterOptions,
} from "./openai-compat.js";
import {
  OpenRouterAdapter,
  type OpenRouterAdapterOptions,
  type OpenRouterProviderPreferences,
} from "./open-router-compat.js";

/**
 * Sources the SDK knows how to build a backend for. Any other string is
 * treated as a generic OpenAI-compatible server and needs a `baseUrl`.
 */
export type KnownProviderSource = "openAI" | "openRouter" | "ollama";

export const KNOWN_PROVIDER_SOURCES: readonly KnownProviderSource[] = [
  "openAI",
  "openRouter",
  "ollama",
] as const;

/** Default endpoint per known source, so config carries only credentials. */
const DEFAULT_BASE_URLS: Record<KnownProviderSource, string> = {
  openAI: "https://api.openai.com/v1",
  openRouter: "https://openrouter.ai/api/v1",
  ollama: "http://localhost:11434/v1",
};

// ---- per-source config ---------------------------------------------------
//
// Each source's config is derived from the options of the adapter that serves
// it, so the two can never drift: a key added to `OpenAICompatAdapterOptions`
// shows up on `openAI` automatically, and a key that adapter doesn't have is a
// type error at the call site instead of a value silently dropped on the floor.
//
// Three deltas are layered on top of the adapter options:
//   1. `baseUrl` becomes optional — every known source has a default endpoint.
//   2. `apiKey` becomes optional in the type, and is checked at construction
//      for the sources that need one.
//   3. Keys belonging to another source's adapter are typed `never`.

/** OpenRouter-only keys, rejected on the sources whose adapters lack them. */
export type OpenRouterOnlyConfig = Pick<
  OpenRouterAdapterOptions,
  "provider" | "usageAccounting" | "appTitle" | "siteUrl"
>;

/** Ollama-only keys: `host` from its adapter, plus the route selector. */
export interface OllamaOnlyConfig {
  /**
   * Server root without the API path, e.g. `"http://localhost:11434"`. `/v1`
   * is appended to reach the OpenAI-compatible route; `baseUrl` wins when both
   * are given. Passed straight through on `api: "native"`.
   */
  host?: string;
  /**
   * Which Ollama route to speak: `"openai"` (default) for
   * `/v1/chat/completions`, `"native"` for NDJSON `/api/chat`.
   */
  api?: "openai" | "native";
}

/** Types a foreign source's keys away, so misplacing one is a compile error. */
type Excluded<T> = { [K in keyof T]?: never };

/**
 * `source: "openAI"` — {@link OpenAICompatAdapterOptions} with a defaulted
 * endpoint (`https://api.openai.com/v1`).
 */
export interface OpenAIProviderConfig
  extends
    Omit<OpenAICompatAdapterOptions, "baseUrl">,
    Excluded<OpenRouterOnlyConfig>,
    Excluded<OllamaOnlyConfig> {
  /** Optional here — defaults to `https://api.openai.com/v1`. */
  baseUrl?: string;
}

/**
 * `source: "openRouter"` — {@link OpenRouterAdapterOptions} verbatim, minus
 * the required `apiKey` (checked when the backend is built). OpenRouter's
 * adapter always sends `max_tokens`, so `maxTokensParam` is not one of its
 * keys.
 */
export interface OpenRouterProviderConfig
  extends Omit<OpenRouterAdapterOptions, "apiKey">, Excluded<OllamaOnlyConfig> {
  /** Optional here — validated when the backend is built. */
  apiKey?: string;
  /** OpenRouter's adapter has no such option. */
  maxTokensParam?: never;
}

/**
 * `source: "ollama"`, default route — the OpenAI-compatible
 * `/v1/chat/completions` Ollama has served since 0.2, so the config is
 * {@link OpenAICompatAdapterOptions} plus `host`.
 */
export interface OllamaCompatProviderConfig
  extends
    Omit<OpenAICompatAdapterOptions, "baseUrl">,
    Excluded<OpenRouterOnlyConfig> {
  /** Optional here — defaults to `http://localhost:11434/v1`. */
  baseUrl?: string;
  /** Server root, e.g. `"http://localhost:11434"`; `/v1` is appended. */
  host?: string;
  api?: "openai";
}

/**
 * `source: "ollama"` with `api: "native"` — the NDJSON `/api/chat` route, so
 * the config is exactly {@link OllamaAdapterOptions}. That adapter speaks a
 * different wire format and takes no bearer token, so the OpenAI-compatible
 * keys are typed away rather than silently ignored.
 */
export interface OllamaNativeProviderConfig
  extends OllamaAdapterOptions, Excluded<OpenRouterOnlyConfig> {
  api: "native";
  /** `/api/chat` takes no bearer token — put auth in `headers` if a gateway needs one. */
  apiKey?: never;
  /** Native `/api/chat` is addressed by `host`, not a versioned base URL. */
  baseUrl?: never;
  /** No `/responses` route on native Ollama. */
  endpoint?: never;
  /** Native bodies carry no token-limit key of either name. */
  maxTokensParam?: never;
}

/** `source: "ollama"` — the OpenAI-compatible route by default, or native NDJSON. */
export type OllamaProviderConfig =
  OllamaCompatProviderConfig | OllamaNativeProviderConfig;

/** The config shape a given source accepts. */
export type ConfigForSource<S extends KnownProviderSource> = S extends "openAI"
  ? OpenAIProviderConfig
  : S extends "openRouter"
    ? OpenRouterProviderConfig
    : OllamaProviderConfig;

/**
 * What one source needs to reach its provider — the permissive superset of
 * every source's keys, used by the `providers` map where the source is a free
 * string. Prefer {@link ConfigForSource} when the source is known.
 */
export interface ProviderConfig
  extends
    Omit<OpenAICompatAdapterOptions, "baseUrl">,
    OpenRouterOnlyConfig,
    OllamaOnlyConfig {
  baseUrl?: string;
}

/**
 * The adapter that fits a source: `openRouter` speaks OpenRouter's dialect
 * (provider routing, usage accounting), `openAI` needs
 * `max_completion_tokens`, `ollama` authenticates however its gateway is
 * configured, and anything else is a plain OpenAI-compatible server.
 *
 * Called by `ResponsesClient` for its own `source`, and exported for driving
 * `AgentLoop` directly. When a source needs something this mapping doesn't
 * express, override `createBackend()` in a `ResponsesClient` subclass and
 * return any `BackendAdapter` — nothing here has to be involved.
 */
export function createBackendForSource(
  source: string,
  config: ProviderConfig = {},
): BackendAdapter {
  const apiKey = config.apiKey;
  const explicitBaseUrl = config.baseUrl;

  // Ollama's native route speaks NDJSON, not the OpenAI schema, so it is
  // chosen before any base-URL resolution.
  if (source === "ollama" && config.api === "native") {
    const host = config.host ?? (explicitBaseUrl && toHost(explicitBaseUrl));
    return new OllamaAdapter({
      ...(host ? { host } : {}),
      ...(config.headers ? { headers: config.headers } : {}),
      ...(config.fetch ? { fetch: config.fetch } : {}),
      ...(config.forceModel ? { forceModel: config.forceModel } : {}),
    });
  }

  const baseUrl =
    explicitBaseUrl ??
    (config.host ? toCompatBaseUrl(config.host) : undefined) ??
    DEFAULT_BASE_URLS[source as KnownProviderSource];
  if (!baseUrl) {
    throw new Error(
      `createBackendForSource: source "${source}" has no built-in endpoint — pass \`baseUrl\` in the config, or override \`createBackend()\` in a ResponsesClient subclass to return an adapter of your own.`,
    );
  }

  if (source === "openRouter") {
    return new OpenRouterAdapter({
      apiKey: apiKey ?? "",
      baseUrl,
      endpoint: config.endpoint ?? "completions",
      ...(config.headers ? { headers: config.headers } : {}),
      ...(config.fetch ? { fetch: config.fetch } : {}),
      ...(config.forceModel ? { forceModel: config.forceModel } : {}),
      ...(config.provider ? { provider: config.provider } : {}),
      ...(config.usageAccounting !== undefined
        ? { usageAccounting: config.usageAccounting }
        : {}),
      ...(config.appTitle ? { appTitle: config.appTitle } : {}),
      ...(config.siteUrl ? { siteUrl: config.siteUrl } : {}),
    });
  }

  return new OpenAICompatAdapter({
    baseUrl,
    // Ollama ignores the key but the OpenAI-compatible route wants one sent.
    apiKey: apiKey ?? (source === "ollama" ? "ollama" : undefined),
    endpoint: config.endpoint ?? "completions",
    maxTokensParam:
      config.maxTokensParam ??
      (source === "openAI" ? "max_completion_tokens" : "max_tokens"),
    ...(config.headers ? { headers: config.headers } : {}),
    ...(config.fetch ? { fetch: config.fetch } : {}),
    ...(config.forceModel ? { forceModel: config.forceModel } : {}),
  });
}

/**
 * A server root to its OpenAI-compatible base URL. A `host` that already
 * carries the version segment is left alone, so `"http://localhost:11434"` and
 * `"http://localhost:11434/v1"` both land on the same endpoint.
 */
function toCompatBaseUrl(host: string): string {
  const trimmed = host.replace(/\/+$/, "");
  return /\/v\d+$/.test(trimmed) ? trimmed : `${trimmed}/v1`;
}

/** The inverse: drops the version segment to get back to the server root. */
function toHost(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "").replace(/\/v\d+$/, "");
}
