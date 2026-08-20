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

describe("OpenAICompatAdapter file input", () => {
  it("rejects a URL in file.file_data before any request is made", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.com/v1",
      fetch: captureFetch(bodies),
    });

    await expect(
      adapter.complete({
        model: "test-model",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "file",
                file: { file_data: "https://cdn.example/contract.pdf" },
              },
            ],
          },
        ],
      }),
    ).rejects.toThrow(/base64 `file_data`/);
    expect(bodies).toHaveLength(0);
  });

  it("rejects a URL in file.file_data on the streaming path too", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.com/v1",
      fetch: captureFetch(bodies),
    });

    const gen = adapter.stream({
      model: "test-model",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "file",
              file: { file_data: "https://cdn.example/contract.pdf" },
            },
          ],
        },
      ],
    });
    await expect(gen.next()).rejects.toThrow(/base64 `file_data`/);
    expect(bodies).toHaveLength(0);
  });

  it("passes base64 data-URI file_data and file_id through", async () => {
    const bodies: Record<string, unknown>[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.com/v1",
      fetch: captureFetch(bodies),
    });

    await adapter.complete({
      model: "test-model",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "file",
              file: { file_data: "data:application/pdf;base64,AAAA" },
            },
            { type: "file", file: { file_id: "file-123" } },
          ],
        },
      ],
    });

    expect(bodies).toHaveLength(1);
  });
});
