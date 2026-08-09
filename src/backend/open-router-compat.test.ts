import { describe, expect, it } from "vitest";
import { OpenRouterAdapter } from "./open-router-compat.js";
import type { ChatCompletionResponse } from "../types/completions.js";

const completion: ChatCompletionResponse = {
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 1,
  model: "test-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "ok" },
      finish_reason: "stop",
    },
  ],
};

const captureFetch = (bodies: Record<string, unknown>[]): typeof fetch =>
  (async (_url: unknown, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return new Response(JSON.stringify(completion), { status: 200 });
  }) as typeof fetch;

const chatRequest = {
  model: "openai/gpt-5-mini",
  messages: [{ role: "user" as const, content: "hi" }],
};

describe("OpenRouterAdapter usage accounting", () => {
  it("injects usage.include on complete() by default", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-key",
      fetch: captureFetch(bodies),
    });

    await adapter.complete(chatRequest);

    expect(bodies[0].usage).toEqual({ include: true });
  });

  it("injects usage.include on stream() by default", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-key",
      fetch: (async (_url: unknown, init?: RequestInit) => {
        bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        return new Response("data: [DONE]\n\n", { status: 200 });
      }) as typeof fetch,
    });

    for await (const _ of adapter.stream(chatRequest)) void _;

    expect(bodies[0].usage).toEqual({ include: true });
    expect(bodies[0].stream_options).toEqual({ include_usage: true });
  });

  it("does not inject when usageAccounting is false", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-key",
      usageAccounting: false,
      fetch: captureFetch(bodies),
    });

    await adapter.complete(chatRequest);

    expect(bodies[0]).not.toHaveProperty("usage");
  });

  it("lets an explicit request usage win over the option", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-key",
      fetch: captureFetch(bodies),
    });

    await adapter.complete({ ...chatRequest, usage: { include: false } });

    expect(bodies[0].usage).toEqual({ include: false });
  });

  it("never injects on embeddings()", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-key",
      fetch: (async (_url: unknown, init?: RequestInit) => {
        bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        return new Response(
          JSON.stringify({
            object: "list",
            data: [{ object: "embedding", index: 0, embedding: [0.1] }],
            model: "baai/bge-m3",
            usage: { prompt_tokens: 1, total_tokens: 1 },
          }),
          { status: 200 },
        );
      }) as typeof fetch,
    });

    await adapter.embeddings({ model: "baai/bge-m3", input: "hi" });

    expect(bodies[0]).not.toHaveProperty("usage");
  });
});
