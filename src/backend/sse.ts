/**
 * Parses a ReadableStream of SSE (Server-Sent Events) data lines into
 * JSON objects. Handles the `data: ` prefix, the `[DONE]` sentinel used by
 * OpenAI-compatible servers, and multi-line events. If an `event:` line is
 * present and the JSON payload is an object without its own `type` field,
 * the event name is injected as `type` so callers dispatching on `type`
 * work with providers that put the event kind in the SSE frame.
 */
export async function* parseSSE<T>(
  body: ReadableStream<Uint8Array> | NodeJS.ReadableStream,
): AsyncGenerator<T> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const iter = toAsyncIterable(body);
  for await (const chunk of iter) {
    buffer += decoder.decode(chunk, { stream: true });
    // Events are separated by a blank line.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseEvent(raw);
      if (parsed === null) {
        // comment or empty event
      } else if (parsed.data === "[DONE]") {
        return;
      } else {
        const value = decodeJson<T>(parsed);
        if (value !== undefined) yield value;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
  // flush
  buffer += decoder.decode();
  const tail = parseEvent(buffer);
  if (tail && tail.data !== "[DONE]") {
    const value = decodeJson<T>(tail);
    if (value !== undefined) yield value;
  }
}

function decodeJson<T>(parsed: {
  event?: string;
  data: string;
}): T | undefined {
  let value: unknown;
  try {
    value = JSON.parse(parsed.data);
  } catch {
    return undefined;
  }
  if (
    parsed.event &&
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as Record<string, unknown>).type === undefined
  ) {
    (value as Record<string, unknown>).type = parsed.event;
  }
  return value as T;
}

function parseEvent(raw: string): { event?: string; data: string } | null {
  const lines = raw.split(/\r?\n/);
  const data: string[] = [];
  let event: string | undefined;
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    } else if (line.startsWith("event:")) {
      event = line.slice(6).replace(/^ /, "");
    }
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

async function* toAsyncIterable(
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
