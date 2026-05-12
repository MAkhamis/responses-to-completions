import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
import type { BackendAdapter } from "./adapter.js";
import { BackendError } from "./openai-compat.js";
import { parseSSE } from "./sse.js";

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/**
 * OpenRouter provider routing preferences.
 * https://openrouter.ai/docs/provider-routing
 */
export interface OpenRouterProviderPreferences {
  /**
   * Provider slugs to try in priority order.
   * e.g. ["Google", "Anthropic"]
   */
  order?: string[];
  /** Fall back to other providers when the preferred ones fail or are unavailable. Default true. */
  allow_fallbacks?: boolean;
  /** Only route to providers that support every parameter sent in the request. */
  require_parameters?: boolean;
  /** Control whether your prompts/completions may be used for training. */
  data_collection?: "allow" | "deny";
  /** Whitelist — only use these provider slugs. */
  only?: string[];
  /** Blacklist — never use these provider slugs. */
  ignore?: string[];
  /** Sort providers by: "price" | "throughput" | "latency". */
  sort?: "price" | "throughput" | "latency";
  /** Quantization level filter: "int4" | "int8" | "fp8" | "fp16" | "bf16" | "unknown". */
  quantizations?: Array<"int4" | "int8" | "fp8" | "fp16" | "bf16" | "unknown">;
}

export interface OpenRouterAdapterOptions {
  apiKey: string;
  /** Defaults to "https://openrouter.ai/api/v1". */
  baseUrl?: string;
  /** Sent as X-Title header for rankings on openrouter.ai. */
  appTitle?: string;
  /** Sent as HTTP-Referer header for rankings on openrouter.ai. */
  siteUrl?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Optional model override: force all requests to use this model regardless of client input. */
  forceModel?: string;
  /** Provider routing preferences applied to every request. */
  provider?: OpenRouterProviderPreferences;
}

export class OpenRouterAdapter implements BackendAdapter {
  readonly name = "openrouter";
  private baseUrl: string;
  private fetch: typeof fetch;

  constructor(private opts: OpenRouterAdapterOptions) {
    this.baseUrl = (opts.baseUrl ?? OPENROUTER_BASE_URL).replace(/\/+$/, "");
    this.fetch = opts.fetch ?? fetch;
  }

  private headers(extra?: Record<string, string>) {
    const h: Record<string, string> = {
      "content-type": "application/json",
      authorization: `Bearer ${this.opts.apiKey}`,
      ...(this.opts.siteUrl ? { "http-referer": this.opts.siteUrl } : {}),
      ...(this.opts.appTitle ? { "x-title": this.opts.appTitle } : {}),
      ...(this.opts.headers ?? {}),
      ...(extra ?? {}),
    };
    return h;
  }

  private prepare(
    req: ChatCompletionRequest,
  ): ChatCompletionRequest & { provider?: OpenRouterProviderPreferences } {
    return {
      ...req,
      ...(this.opts.forceModel ? { model: this.opts.forceModel } : {}),
      ...(this.opts.provider ? { provider: this.opts.provider } : {}),
    };
  }

  async complete(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const res = await this.fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ ...this.prepare(req), stream: false }),
      signal,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new BackendError(res.status, body);
    }
    return (await res.json()) as ChatCompletionResponse;
  }

  async *stream(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    const res = await this.fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers({ accept: "text/event-stream" }),
      body: JSON.stringify({
        ...this.prepare(req),
        stream: true,
        stream_options: { include_usage: true, ...(req.stream_options ?? {}) },
      }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = res.body ? await res.text() : "(no body)";
      throw new BackendError(res.status, body);
    }
    for await (const ev of parseSSE<ChatCompletionChunk>(res.body)) {
      yield ev;
    }
  }
}
