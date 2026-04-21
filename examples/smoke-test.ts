/**
 * Smoke test: stubs the backend fetch and the MCP client, then drives one
 * non-streaming and one streaming request through the server in-process.
 *
 *   npx tsx examples/smoke-test.ts
 */
import { createServer, OpenAICompatAdapter, LocalFileStore } from "../src/index.js";
import type { ChatCompletionChunk, ChatCompletionResponse } from "../src/types/completions.js";
import * as path from "node:path";
import * as os from "node:os";
import { promises as fs } from "node:fs";

async function main() {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "r2c-smoke-"));

  // Stub backend responses
  const fakeFetch: typeof fetch = async (input, init) => {
    const url = String(input);
    const body = JSON.parse(String(init?.body ?? "{}"));
    console.log(`[fake fetch] ${init?.method} ${url} stream=${body.stream}`);
    if (body.stream) {
      const chunks: ChatCompletionChunk[] = [
        { id: "c1", object: "chat.completion.chunk", created: 1, model: body.model,
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] },
        { id: "c1", object: "chat.completion.chunk", created: 1, model: body.model,
          choices: [{ index: 0, delta: { content: "Hello " }, finish_reason: null }] },
        { id: "c1", object: "chat.completion.chunk", created: 1, model: body.model,
          choices: [{ index: 0, delta: { content: "world" }, finish_reason: null }] },
        { id: "c1", object: "chat.completion.chunk", created: 1, model: body.model,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } },
      ];
      const sse = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n";
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        },
      });
      return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
    }
    const resp: ChatCompletionResponse = {
      id: "c1", object: "chat.completion", created: 1, model: body.model,
      choices: [{ index: 0, message: { role: "assistant", content: "Hi there!" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    };
    return new Response(JSON.stringify(resp), { status: 200, headers: { "content-type": "application/json" } });
  };

  const app = createServer({
    backend: new OpenAICompatAdapter({
      baseUrl: "http://stub.invalid/v1",
      fetch: fakeFetch,
    }),
    store: new LocalFileStore(tmp),
  });

  const server = app.listen(0);
  const port = (server.address() as { port: number }).port;
  const base = `http://127.0.0.1:${port}`;
  console.log("booted on", base);

  // --- Non-streaming ---
  const resp1 = await fetch(`${base}/v1/responses`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "llama3.1", input: "Say hi" }),
  });
  const j1 = await resp1.json();
  console.log("non-streaming response:", JSON.stringify({
    id: j1.id, status: j1.status, output_text: j1.output_text, outputN: j1.output.length,
  }));

  // --- Streaming ---
  const resp2 = await fetch(`${base}/v1/responses`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "llama3.1", input: "Say hi again", stream: true }),
  });
  console.log("streaming status:", resp2.status, resp2.headers.get("content-type"));
  const reader = resp2.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const events: string[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, nl);
      buf = buf.slice(nl + 2);
      const firstLine = raw.split("\n")[0];
      if (firstLine.startsWith("event: ")) events.push(firstLine.slice(7));
    }
  }
  console.log("streaming events:", events);

  // --- Conversation continuation ---
  const conv = await fetch(`${base}/v1/conversations`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ metadata: { app: "smoke" } }),
  }).then((r) => r.json());
  console.log("created conv:", conv.id);

  await fetch(`${base}/v1/responses`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "llama3.1", input: "msg 1", conversation: conv.id }),
  }).then((r) => r.json());
  await fetch(`${base}/v1/responses`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "llama3.1", input: "msg 2", conversation: conv.id }),
  }).then((r) => r.json());

  const items = await fetch(`${base}/v1/conversations/${conv.id}/items`).then((r) => r.json());
  console.log("items after 2 turns:", items.data.length, "entries");

  server.close();
  await fs.rm(tmp, { recursive: true, force: true });
  console.log("OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
