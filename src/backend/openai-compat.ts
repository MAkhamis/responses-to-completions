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
import { parseSSE } from "./sse.js";

export interface OpenAICompatAdapterOptions {
  /** Base URL without trailing endpoint, e.g. "https://api.openai.com/v1". */
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Optional model override: force all requests to use this model regardless of client input. */
  forceModel?: string;
  /**
   * Which upstream endpoint to call:
   *  - `"completions"` (default) → `POST {baseUrl}/chat/completions`
   *  - `"responses"`             → `POST {baseUrl}/responses`
   */
  endpoint?: "completions" | "responses";
  /**
   * Which token-limit key to send on chat-completions requests.
   * api.openai.com rejects `max_tokens` for reasoning models (o-series,
   * gpt-5*) and wants `max_completion_tokens`; most OSS servers only know
   * `max_tokens`. Default `"max_tokens"`.
   */
  maxTokensParam?: "max_tokens" | "max_completion_tokens";
}

/**
 * Talks to any server implementing either of OpenAI's HTTP surfaces:
 *
 * - `endpoint: "completions"` (default) — `POST /chat/completions` with the
 *   OpenAI Chat-Completions JSON schema. Covers OpenAI, Ollama's
 *   `/v1/chat/completions`, vLLM, llama.cpp server, TGI with openai-adapter,
 *   LiteLLM, Together, Groq.
 *
 * - `endpoint: "responses"` — `POST /responses` with the OpenAI
 *   Responses-API schema. Pass-through with no translation.
 */
export class OpenAICompatAdapter implements BackendAdapter {
  readonly name = "openai-compat";
  readonly mode: "completions" | "responses";
  private baseUrl: string;
  private fetch: typeof fetch;

  constructor(private opts: OpenAICompatAdapterOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.fetch = opts.fetch ?? fetch;
    this.mode = opts.endpoint ?? "completions";
  }

  private headers(extra?: Record<string, string>) {
    const h: Record<string, string> = {
      "content-type": "application/json",
      ...(this.opts.headers ?? {}),
      ...(extra ?? {}),
    };
    if (this.opts.apiKey) h["authorization"] = `Bearer ${this.opts.apiKey}`;
    return h;
  }

  private prepare<T extends { model: string }>(req: T): T {
    let out: T = this.opts.forceModel
      ? { ...req, model: this.opts.forceModel }
      : req;
    const maxTokens = (out as { max_tokens?: number }).max_tokens;
    if (
      this.opts.maxTokensParam === "max_completion_tokens" &&
      maxTokens !== undefined
    ) {
      const { max_tokens: _max_tokens, ...rest } = out as T & {
        max_tokens?: number;
      };
      out = { ...rest, max_completion_tokens: maxTokens } as unknown as T;
    }
    return out;
  }

  // ---- chat-completions endpoint ------------------------------------------
  async complete(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    assertNoUrlFileData(req);
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
    assertNoUrlFileData(req);
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
 * Chat-completions backends behind this adapter take documents as base64
 * `file_data` or an uploaded `file_id` only — a plain URL in `file_data` is
 * an OpenRouter file-parser extension (the translator maps
 * `input_file.file_url` there for `OpenRouterAdapter` to consume).
 */
function assertNoUrlFileData(req: ChatCompletionRequest): void {
  for (const msg of req.messages) {
    if (!Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (part.type !== "file") continue;
      const data = part.file.file_data;
      if (data && /^https?:\/\//i.test(data)) {
        throw new Error(
          `input_file: this backend takes documents as base64 \`file_data\` or an uploaded \`file_id\`, not a URL ("${data}"). Inline the file as base64 \`file_data\`, upload it and pass \`file_id\`, or use a backend that accepts URLs — source "openRouter", or OpenAI's Responses endpoint via \`endpoint: "responses"\`.`,
        );
      }
    }
  }
}

export class BackendError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`Backend error ${status}: ${body.slice(0, 500)}`);
    this.name = "BackendError";
  }
}
