import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
import type { BackendAdapter } from "./adapter.js";
import { parseSSE } from "./sse.js";

export interface OpenAICompatAdapterOptions {
  /** Base URL without trailing /chat/completions, e.g. "http://localhost:8000/v1". */
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Optional model override: force all requests to use this model regardless of client input. */
  forceModel?: string;
}

/**
 * Works with any server implementing POST /chat/completions with OpenAI's
 * JSON schema. Covers: OpenAI itself, Ollama's /v1/chat/completions,
 * vLLM, llama.cpp server, TGI with openai-adapter, LiteLLM, Together, Groq.
 */
export class OpenAICompatAdapter implements BackendAdapter {
  readonly name = "openai-compat";
  private baseUrl: string;
  private fetch: typeof fetch;

  constructor(private opts: OpenAICompatAdapterOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.fetch = opts.fetch ?? fetch;
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

  private prepare(req: ChatCompletionRequest): ChatCompletionRequest {
    if (this.opts.forceModel) return { ...req, model: this.opts.forceModel };
    return req;
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

export class BackendError extends Error {
  constructor(public status: number, public body: string) {
    super(`Backend error ${status}: ${body.slice(0, 500)}`);
    this.name = "BackendError";
  }
}
