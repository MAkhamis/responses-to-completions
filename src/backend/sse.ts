/**
 * Parses a ReadableStream of SSE (Server-Sent Events) data lines into
 * JSON objects. Handles the `data: ` prefix, the `[DONE]` sentinel used by
 * OpenAI-compatible servers, and multi-line events.
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
      const data = parseEvent(raw);
      if (data === null) {
        // comment or empty event
      } else if (data === "[DONE]") {
        return;
      } else {
        try {
          yield JSON.parse(data) as T;
        } catch {
          // skip malformed
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
  // flush
  buffer += decoder.decode();
  const tail = parseEvent(buffer);
  if (tail && tail !== "[DONE]") {
    try {
      yield JSON.parse(tail) as T;
    } catch {
      // ignore
    }
  }
}

function parseEvent(raw: string): string | null {
  const lines = raw.split(/\r?\n/);
  const data: string[] = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (data.length === 0) return null;
  return data.join("\n");
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
      yield typeof chunk === "string" ? new TextEncoder().encode(chunk) : (chunk as Uint8Array);
    }
  }
}
