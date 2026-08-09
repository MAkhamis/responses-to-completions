import { describe, expect, it } from "vitest";
import { OpenAICompatAdapter } from "./openai-compat.js";
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

describe("OpenAICompatAdapter maxTokensParam", () => {
  it("sends max_tokens by default", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://example.test/v1",
      fetch: captureFetch(bodies),
    });

    await adapter.complete({
      model: "test-model",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
    });

    expect(bodies[0].max_tokens).toBe(100);
    expect(bodies[0]).not.toHaveProperty("max_completion_tokens");
  });

  it("renames max_tokens to max_completion_tokens when configured", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://example.test/v1",
      fetch: captureFetch(bodies),
      maxTokensParam: "max_completion_tokens",
    });

    await adapter.complete({
      model: "test-model",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
    });

    expect(bodies[0].max_completion_tokens).toBe(100);
    expect(bodies[0]).not.toHaveProperty("max_tokens");
  });

  it("does not add max_completion_tokens when no limit was requested", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://example.test/v1",
      fetch: captureFetch(bodies),
      maxTokensParam: "max_completion_tokens",
    });

    await adapter.complete({
      model: "test-model",
      messages: [{ role: "user", content: "hi" }],
    });

    expect(bodies[0]).not.toHaveProperty("max_tokens");
    expect(bodies[0]).not.toHaveProperty("max_completion_tokens");
  });

  it("keeps forceModel working together with the rename", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://example.test/v1",
      fetch: captureFetch(bodies),
      forceModel: "forced-model",
      maxTokensParam: "max_completion_tokens",
    });

    await adapter.complete({
      model: "test-model",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 50,
    });

    expect(bodies[0].model).toBe("forced-model");
    expect(bodies[0].max_completion_tokens).toBe(50);
    expect(bodies[0]).not.toHaveProperty("max_tokens");
  });
});
