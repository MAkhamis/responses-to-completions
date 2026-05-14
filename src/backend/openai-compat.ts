import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
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
    if (this.opts.forceModel) return { ...req, model: this.opts.forceModel };
    return req;
  }

  // ---- chat-completions endpoint ------------------------------------------
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
