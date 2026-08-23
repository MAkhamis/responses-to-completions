import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
import type {
  EmbeddingsRequest,
  EmbeddingsResponse,
} from "../types/embeddings.js";
import type {
  CreateResponseRequest,
  ResponseObject,
} from "../types/responses.js";
import type { StreamEvent } from "../translate/stream.js";
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
  /**
   * Which upstream endpoint to call:
   *  - `"completions"` (default) → `POST {baseUrl}/chat/completions`
   *  - `"responses"`             → `POST {baseUrl}/responses`
   */
  endpoint?: "completions" | "responses";
  /**
   * OpenRouter usage accounting: injects `usage: { include: true }` into
   * chat-completions requests so responses report the billed `usage.cost`
   * (credits). Default true; set false to opt out (OpenRouter computes the
   * cost before emitting the final chunk, which can add a small delay). An
   * explicit `usage` field on a request always wins over this option.
   * Applies to /chat/completions only — never to /responses or /embeddings.
   */
  usageAccounting?: boolean;
}

export class OpenRouterAdapter implements BackendAdapter {
  readonly name = "openrouter";
  readonly mode: "completions" | "responses";
  private baseUrl: string;
  private fetch: typeof fetch;

  constructor(private opts: OpenRouterAdapterOptions) {
    this.baseUrl = (opts.baseUrl ?? OPENROUTER_BASE_URL).replace(/\/+$/, "");
    this.fetch = opts.fetch ?? fetch;
    this.mode = opts.endpoint ?? "completions";
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

  private prepare<T extends { model: string }>(
    req: T,
  ): T & { provider?: OpenRouterProviderPreferences } {
    return {
      ...req,
      ...(this.opts.forceModel ? { model: this.opts.forceModel } : {}),
      ...(this.opts.provider ? { provider: this.opts.provider } : {}),
    };
  }

  /** Usage-accounting opt-in for chat-completions bodies (see option docs). */
  private usageBody(req: ChatCompletionRequest): {
    usage?: { include?: boolean };
  } {
    return req.usage === undefined && this.opts.usageAccounting !== false
      ? { usage: { include: true } }
      : {};
  }

  // ---- chat-completions endpoint ------------------------------------------
  async complete(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    assertNoFileId(req);
    const res = await this.fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        ...this.prepare(req),
        ...this.usageBody(req),
        stream: false,
      }),
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
    assertNoFileId(req);
    const res = await this.fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.headers({ accept: "text/event-stream" }),
      body: JSON.stringify({
        ...this.prepare(req),
        ...this.usageBody(req),
        stream: true,
        stream_options: { include_usage: true, ...(req.stream_options ?? {}) },
      }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "(no body)");
      throw new BackendError(res.status, body);
    }
    for await (const ev of parseSSE<ChatCompletionChunk>(res.body)) {
      yield ev;
    }
  }

  // ---- responses endpoint -------------------------------------------------
  async respond(
    req: CreateResponseRequest,
    signal?: AbortSignal,
  ): Promise<ResponseObject> {
    const res = await this.fetch(`${this.baseUrl}/responses`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ ...this.prepare(req), stream: false }),
      signal,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new BackendError(res.status, body);
    }
    return (await res.json()) as ResponseObject;
  }

  async *respondStream(
    req: CreateResponseRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent> {
    const res = await this.fetch(`${this.baseUrl}/responses`, {
      method: "POST",
      headers: this.headers({ accept: "text/event-stream" }),
      body: JSON.stringify({ ...this.prepare(req), stream: true }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "(no body)");
      throw new BackendError(res.status, body);
    }
    for await (const ev of parseSSE<StreamEvent>(res.body)) {
      yield ev;
    }
  }

  async embeddings(
    req: EmbeddingsRequest,
    signal?: AbortSignal,
  ): Promise<EmbeddingsResponse> {
    const res = await this.fetch(`${this.baseUrl}/embeddings`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(this.prepare(req)),
      signal,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new BackendError(res.status, body);
    }
    return (await res.json()) as EmbeddingsResponse;
  }
}

/**
 * A URL in `file_data` is fine here — that is OpenRouter's file-parser
 * extension — but `file_id` names an upload hosted by another provider, which
 * OpenRouter cannot resolve. This adapter is an independent implementation
 * rather than a subclass of `OpenAICompatAdapter`, so it needs its own check;
 * without one, forwarding a `file_id` yields the opaque upstream 400 the
 * sibling adapter's guard exists to replace.
 */
function assertNoFileId(req: ChatCompletionRequest): void {
  for (const msg of req.messages) {
    if (!Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (part.type !== "file") continue;
      if (part.file.file_id) {
        throw new Error(
          `input_file: source "openRouter" cannot resolve \`file_id\` ("${part.file.file_id}") — that names an upload hosted by another provider. Pass the document as base64 \`file_data\`, or as a URL in \`file_url\` for OpenRouter's file parser.`,
        );
      }
    }
  }
}
