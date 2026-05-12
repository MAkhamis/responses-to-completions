import type {
  ChatCompletionChunk,
  ChatCompletionChunkChoice,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatToolCall,
} from "../types/completions.js";
import type { BackendAdapter } from "./adapter.js";
import { BackendError } from "./openai-compat.js";

export interface OllamaAdapterOptions {
  /** Host like "http://localhost:11434". No trailing slash needed. */
  host?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  forceModel?: string;
}

/**
 * Native Ollama /api/chat adapter.
 *
 * Differences vs OpenAI spec that we normalize here:
 *  - Ollama streams newline-delimited JSON (NDJSON), not SSE.
 *  - tool_calls[].function.arguments is an OBJECT, not a JSON string.
 *  - `done: true` carries final stats; there is no per-choice finish_reason,
 *    we synthesize one.
 *  - Roles are limited to system/user/assistant/tool; no `developer`.
 *
 * In practice Ollama >=0.2 also exposes /v1/chat/completions (OpenAI-compatible),
 * so prefer OpenAICompatAdapter unless you need NDJSON or native-only features.
 */
export class OllamaAdapter implements BackendAdapter {
  readonly name = "ollama";
  private host: string;
  private fetch: typeof fetch;

  constructor(private opts: OllamaAdapterOptions = {}) {
    this.host = (opts.host ?? "http://localhost:11434").replace(/\/+$/, "");
    this.fetch = opts.fetch ?? fetch;
  }

  private headers() {
    return {
      "content-type": "application/json",
      ...(this.opts.headers ?? {}),
    };
  }

  private toOllamaBody(req: ChatCompletionRequest, stream: boolean) {
    const model = this.opts.forceModel ?? req.model;
    const messages = req.messages.map((m) => {
      if (m.role === "tool") {
        return {
          role: "tool",
          content:
            typeof m.content === "string" ? m.content : partsToText(m.content),
          tool_call_id: m.tool_call_id,
        };
      }
      const asst = m as typeof m & { tool_calls?: ChatToolCall[] };
      return {
        role: m.role === "developer" ? "system" : m.role,
        content:
          typeof m.content === "string" ? m.content : partsToText(m.content),
        ...(asst.tool_calls
          ? {
              tool_calls: asst.tool_calls.map((tc) => ({
                function: {
                  name: tc.function.name,
                  arguments: safeJsonParse(tc.function.arguments),
                },
              })),
            }
          : {}),
      };
    });

    const options: Record<string, unknown> = {};
    if (req.temperature !== undefined) options.temperature = req.temperature;
    if (req.top_p !== undefined) options.top_p = req.top_p;
    if (req.max_tokens !== undefined) options.num_predict = req.max_tokens;
    if (req.max_completion_tokens !== undefined)
      options.num_predict = req.max_completion_tokens;
    if (req.stop !== undefined)
      options.stop = Array.isArray(req.stop) ? req.stop : [req.stop];
    if (req.seed !== undefined) options.seed = req.seed;

    return {
      model,
      messages,
      stream,
      ...(req.tools ? { tools: req.tools } : {}),
      ...(Object.keys(options).length ? { options } : {}),
      ...(req.response_format?.type === "json_object"
        ? { format: "json" }
        : {}),
      ...(req.response_format?.type === "json_schema"
        ? { format: req.response_format.json_schema.schema }
        : {}),
    };
  }

  async complete(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse> {
    const res = await this.fetch(`${this.host}/api/chat`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(this.toOllamaBody(req, false)),
      signal,
    });
    if (!res.ok) throw new BackendError(res.status, await res.text());
    const body = (await res.json()) as OllamaChatResponse;
    return ollamaToOpenAI(body);
  }

  async *stream(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    const res = await this.fetch(`${this.host}/api/chat`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(this.toOllamaBody(req, true)),
      signal,
    });
    if (!res.ok || !res.body)
      throw new BackendError(res.status, res.body ? await res.text() : "");

    const id = `chatcmpl-ollama-${Date.now().toString(36)}`;
    const created = Math.floor(Date.now() / 1000);
    const model = this.opts.forceModel ?? req.model;
    let sentRole = false;
    let toolCallEmitted = false;

    for await (const ev of parseNdjson<OllamaChatResponse>(res.body)) {
      const choices: ChatCompletionChunkChoice[] = [];
      const delta: ChatCompletionChunkChoice["delta"] = {};

      if (!sentRole) {
        delta.role = "assistant";
        sentRole = true;
      }
      if (ev.message?.content) delta.content = ev.message.content;
      if (ev.message?.tool_calls?.length) {
        delta.tool_calls = ev.message.tool_calls.map((tc, index) => ({
          index,
          id: (tc as { id?: string }).id ?? `call_${id}_${index}`,
          type: "function",
          function: {
            name: tc.function.name,
            arguments:
              typeof tc.function.arguments === "string"
                ? tc.function.arguments
                : JSON.stringify(tc.function.arguments ?? {}),
          },
        }));
        toolCallEmitted = true;
      }

      const finish_reason = ev.done
        ? toolCallEmitted
          ? "tool_calls"
          : ev.done_reason === "length"
            ? "length"
            : "stop"
        : null;

      choices.push({ index: 0, delta, finish_reason });

      const chunk: ChatCompletionChunk = {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices,
      };

      if (ev.done && ev.prompt_eval_count !== undefined) {
        chunk.usage = {
          prompt_tokens: ev.prompt_eval_count ?? 0,
          completion_tokens: ev.eval_count ?? 0,
          total_tokens: (ev.prompt_eval_count ?? 0) + (ev.eval_count ?? 0),
        };
      }
      yield chunk;
      if (ev.done) return;
    }
  }
}

// ---- Ollama wire types ----
interface OllamaChatResponse {
  model: string;
  created_at: string;
  message?: {
    role: string;
    content: string;
    tool_calls?: Array<{
      function: { name: string; arguments: Record<string, unknown> | string };
    }>;
  };
  done: boolean;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
  total_duration?: number;
}

function ollamaToOpenAI(r: OllamaChatResponse): ChatCompletionResponse {
  const toolCalls: ChatToolCall[] | undefined = r.message?.tool_calls?.map(
    (tc, i) => ({
      id: `call_${Date.now().toString(36)}_${i}`,
      type: "function",
      function: {
        name: tc.function.name,
        arguments:
          typeof tc.function.arguments === "string"
            ? tc.function.arguments
            : JSON.stringify(tc.function.arguments ?? {}),
      },
    }),
  );

  const finish_reason = toolCalls?.length
    ? "tool_calls"
    : r.done_reason === "length"
      ? "length"
      : "stop";

  return {
    id: `chatcmpl-ollama-${Date.now().toString(36)}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: r.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: r.message?.content ?? "",
          ...(toolCalls ? { tool_calls: toolCalls } : {}),
        },
        finish_reason,
      },
    ],
    usage: {
      prompt_tokens: r.prompt_eval_count ?? 0,
      completion_tokens: r.eval_count ?? 0,
      total_tokens: (r.prompt_eval_count ?? 0) + (r.eval_count ?? 0),
    },
  };
}

function partsToText(parts: unknown): string {
  if (!Array.isArray(parts)) return String(parts ?? "");
  return parts
    .map((p) =>
      typeof p === "object" && p && "text" in p
        ? (p as { text: string }).text
        : "",
    )
    .join("");
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}

async function* parseNdjson<T>(
  body: ReadableStream<Uint8Array> | NodeJS.ReadableStream,
): AsyncGenerator<T> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const iter = toAsync(body);
  for await (const chunk of iter) {
    buffer += decoder.decode(chunk, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      try {
        yield JSON.parse(line) as T;
      } catch {
        // skip
      }
    }
  }
  const tail = buffer.trim();
  if (tail) {
    try {
      yield JSON.parse(tail) as T;
    } catch {
      // ignore
    }
  }
}

async function* toAsync(
  body: ReadableStream<Uint8Array> | NodeJS.ReadableStream,
): AsyncGenerator<Uint8Array> {
  if (typeof (body as ReadableStream).getReader === "function") {
    const reader = (body as ReadableStream<Uint8Array>).getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        if (value) yield value;
      }
    } finally {
      reader.releaseLock();
    }
  } else {
    for await (const chunk of body as NodeJS.ReadableStream) {
      yield typeof chunk === "string"
        ? new TextEncoder().encode(chunk)
        : (chunk as Uint8Array);
    }
  }
}
