import { describe, expect, it } from "vitest";
import { AgentLoop, mergeUsage } from "../agent-loop.js";
import type { BackendAdapter } from "../backend/adapter.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
import type { ResponseObject } from "../types/responses.js";
import { translateServerTools } from "./request.js";
import { completionToOutputItems, translateUsage } from "./response.js";
import { translateChunkStream } from "./stream.js";

// Shapes below are trimmed from real OpenRouter responses (2026-09-23,
// openai/gpt-6-luna + openrouter:web_search): citations nest under
// `url_citation` with 0 offsets, and usage reports server_tool_use_details
// with the search fee already inside `cost`.
const citation = (url: string, title: string) => ({
  type: "url_citation" as const,
  url_citation: { url, title, start_index: 0, end_index: 0, content: "…" },
});

const orUsage = {
  prompt_tokens: 1721,
  completion_tokens: 130,
  total_tokens: 1851,
  cost: 0.007270525,
  server_tool_use_details: {
    web_search_requests: 1,
    tool_calls_requested: 1,
    tool_calls_executed: 1,
  },
};

describe("translateServerTools", () => {
  it("forwards OpenRouter server tools verbatim on OpenRouter", () => {
    expect(
      translateServerTools(
        [
          { type: "function", name: "f", parameters: {} },
          { type: "openrouter:web_search", parameters: { max_results: 3 } },
          { type: "openrouter:web_fetch" },
          { type: "mcp", server_label: "x", server_url: "https://x" },
        ],
        "openrouter",
      ),
    ).toEqual([
      { type: "openrouter:web_search", parameters: { max_results: 3 } },
      { type: "openrouter:web_fetch" },
    ]);
  });

  it("maps OpenAI's hosted web_search onto openrouter:web_search", () => {
    expect(
      translateServerTools(
        [
          {
            type: "web_search",
            search_context_size: "low",
            filters: { allowed_domains: ["istd.gov.jo"] },
            user_location: { type: "approximate", country: "JO", city: null },
          },
        ],
        "openrouter",
      ),
    ).toEqual([
      {
        type: "openrouter:web_search",
        parameters: {
          search_context_size: "low",
          allowed_domains: ["istd.gov.jo"],
          user_location: { country: "JO" },
        },
      },
    ]);
    expect(translateServerTools([{ type: "web_search" }], "openrouter")).toEqual(
      [{ type: "openrouter:web_search" }],
    );
  });

  it("refuses provider tools on backends that cannot run them", () => {
    expect(() =>
      translateServerTools([{ type: "web_search" }], "openai-compat"),
    ).toThrow(/endpoint: "responses"/);
    expect(() =>
      translateServerTools([{ type: "openrouter:web_fetch" }], "ollama"),
    ).toThrow(/source "openRouter"/);
    expect(translateServerTools(undefined, "ollama")).toEqual([]);
  });
});

describe("citations and server tool usage (non-streaming)", () => {
  it("turns message annotations into url_citation annotations", () => {
    const resp: ChatCompletionResponse = {
      id: "c1",
      object: "chat.completion",
      created: 1,
      model: "openai/gpt-6-luna",
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: {
            role: "assistant",
            content: "Jordan's general sales tax is 16%.",
            annotations: [
              citation("https://www.istd.gov.jo/EN/List/General__Sales_Tax_Tables", "GST tables"),
            ],
          },
        },
      ],
      usage: orUsage,
    };
    const { items } = completionToOutputItems(resp);
    expect(items[0]).toMatchObject({
      type: "message",
      content: [
        {
          type: "output_text",
          text: "Jordan's general sales tax is 16%.",
          annotations: [
            {
              type: "url_citation",
              url: "https://www.istd.gov.jo/EN/List/General__Sales_Tax_Tables",
              title: "GST tables",
              start_index: 0,
              end_index: 0,
              content: "…",
            },
          ],
        },
      ],
    });
  });

  it("carries server_tool_use_details and sums them across iterations", () => {
    const usage = translateUsage(orUsage)!;
    expect(usage.server_tool_use_details).toEqual({
      web_search_requests: 1,
      tool_calls_requested: 1,
      tool_calls_executed: 1,
    });
    expect(usage.cost).toBe(0.007270525);
    const merged = mergeUsage(usage, usage)!;
    expect(merged.server_tool_use_details?.web_search_requests).toBe(2);
    expect(mergeUsage(translateUsage({ prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }), null)).not.toHaveProperty(
      "server_tool_use_details",
    );
  });
});

describe("citations while streaming", () => {
  const snapshot = {
    id: "resp_1",
    object: "response",
    model: "openai/gpt-6-luna",
  } as unknown as ResponseObject;

  const chunk = (
    delta: ChatCompletionChunk["choices"][0]["delta"],
    finish: ChatCompletionChunk["choices"][0]["finish_reason"] = null,
    usage?: ChatCompletionChunk["usage"],
  ): ChatCompletionChunk => ({
    id: "c1",
    object: "chat.completion.chunk",
    created: 1,
    model: "openai/gpt-6-luna",
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage ? { usage } : {}),
  });

  async function* chunks() {
    yield chunk({ role: "assistant", content: "Jordan's rate is 16%." });
    yield chunk({ role: "assistant", annotations: [citation("https://a.example/1", "A")] });
    yield chunk({ role: "assistant", annotations: [citation("https://b.example/2", "B")] });
    yield chunk({ role: "assistant" }, "stop");
    yield chunk({ role: "assistant" }, "stop", orUsage);
  }

  it("emits annotation events and keeps the citations on the final message", async () => {
    const gen = translateChunkStream(chunks(), snapshot);
    const events = [];
    let result;
    for (;;) {
      const r = await gen.next();
      if (r.done) {
        result = r.value;
        break;
      }
      events.push(r.value);
    }
    const added = events.filter(
      (e) => e.type === "response.output_text.annotation.added",
    );
    expect(added.map((e: any) => [e.annotation_index, e.annotation.url])).toEqual([
      [0, "https://a.example/1"],
      [1, "https://b.example/2"],
    ]);
    const partDone: any = events.find((e) => e.type === "response.content_part.done");
    expect(partDone.part.annotations).toHaveLength(2);
    const message: any = result.items.find((i: any) => i.type === "message");
    expect(message.content[0].text).toBe("Jordan's rate is 16%.");
    expect(message.content[0].annotations.map((a: any) => a.title)).toEqual(["A", "B"]);
    expect(result.usage?.server_tool_use_details?.web_search_requests).toBe(1);
  });
});

describe("AgentLoop forwards server tools to OpenRouter's chat completions", () => {
  const completion: ChatCompletionResponse = {
    id: "c1",
    object: "chat.completion",
    created: 1,
    model: "openai/gpt-6-luna",
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: {
          role: "assistant",
          content: "16%.",
          annotations: [citation("https://a.example/1", "A")],
        },
      },
    ],
    usage: orUsage,
  };

  it("sends the mapped tool and returns citations + usage", async () => {
    const captured: ChatCompletionRequest[] = [];
    const backend: BackendAdapter = {
      name: "openrouter",
      mode: "completions",
      complete: async (req) => {
        captured.push(req);
        return completion;
      },
    };
    const result = await new AgentLoop({ backend }).run({
      request: {
        model: "openai/gpt-6-luna",
        input: "Jordan VAT?",
        tools: [{ type: "web_search", search_context_size: "medium" }],
      },
      history: [],
    });
    expect(captured[0].tools).toEqual([
      {
        type: "openrouter:web_search",
        parameters: { search_context_size: "medium" },
      },
    ]);
    const message: any = result.items.find((i) => i.type === "message");
    expect(message.content[0].annotations[0].url).toBe("https://a.example/1");
    expect(result.usage?.server_tool_use_details?.web_search_requests).toBe(1);
  });

  it("refuses a hosted web_search on a backend that cannot run it", async () => {
    const backend: BackendAdapter = {
      name: "openai-compat",
      mode: "completions",
      complete: async () => completion,
    };
    await expect(
      new AgentLoop({ backend }).run({
        request: { model: "gpt", input: "hi", tools: [{ type: "web_search" }] },
        history: [],
      }),
    ).rejects.toThrow(/cannot run provider tools/);
  });
});
