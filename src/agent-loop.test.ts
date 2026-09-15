import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentLoop, mergeUsage } from "./agent-loop.js";
import type { BackendAdapter } from "./backend/adapter.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "./types/completions.js";
import type { CreateResponseRequest } from "./types/responses.js";

const completion = (text = "ok"): ChatCompletionResponse => ({
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 1,
  model: "test-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

const captureBackend = (captured: ChatCompletionRequest[]): BackendAdapter => ({
  name: "capture",
  mode: "completions",
  complete: async (req) => {
    captured.push(req);
    return completion();
  },
});

describe("buildChatRequest translation (via AgentLoop.run)", () => {
  it("translates reasoning.effort to reasoning_effort and drops the reasoning object", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: captureBackend(captured) });

    await agent.run({
      request: {
        model: "test-model",
        input: "hi",
        reasoning: { effort: "minimal", summary: "auto" },
      },
      history: [],
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].reasoning_effort).toBe("minimal");
    expect(captured[0]).not.toHaveProperty("reasoning");
  });

  it("omits reasoning_effort when reasoning has no effort", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: captureBackend(captured) });

    await agent.run({
      request: {
        model: "test-model",
        input: "hi",
        reasoning: { summary: "auto" },
      },
      history: [],
    });

    expect(captured[0]).not.toHaveProperty("reasoning_effort");
    expect(captured[0]).not.toHaveProperty("reasoning");
  });

  it("forwards service_tier to the chat-completions request", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: captureBackend(captured) });

    await agent.run({
      request: { model: "test-model", input: "hi", service_tier: "flex" },
      history: [],
    });

    expect(captured[0].service_tier).toBe("flex");
  });

  it("leaves service_tier off the request when the client didn't set it", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: captureBackend(captured) });

    await agent.run({
      request: { model: "test-model", input: "hi" },
      history: [],
    });

    expect(captured[0]).not.toHaveProperty("service_tier");
  });

  it("carries the backend's billed cost into the run usage", async () => {
    const backend: BackendAdapter = {
      name: "cost-backend",
      mode: "completions",
      complete: async () => ({
        ...completion(),
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cost: 0.00123,
          cost_details: {
            upstream_inference_prompt_cost: 0.0005,
            upstream_inference_completions_cost: 0.00073,
          },
        },
      }),
    };
    const agent = new AgentLoop({ backend });

    const result = await agent.run({
      request: { model: "test-model", input: "hi" },
      history: [],
    });

    expect(result.usage).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cost: 0.00123,
      cost_details: {
        upstream_inference_prompt_cost: 0.0005,
        upstream_inference_completions_cost: 0.00073,
      },
    });
  });
});

describe("mergeUsage token details", () => {
  it("sums cache reads and writes across iterations", () => {
    const merged = mergeUsage(
      {
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        input_tokens_details: { cached_tokens: 0, cache_write_tokens: 96 },
        output_tokens_details: { reasoning_tokens: 4 },
      },
      {
        input_tokens: 120,
        output_tokens: 8,
        total_tokens: 128,
        input_tokens_details: { cached_tokens: 96, cache_write_tokens: 16 },
        output_tokens_details: { reasoning_tokens: 2 },
      },
    );

    expect(merged).toEqual({
      input_tokens: 220,
      output_tokens: 18,
      total_tokens: 238,
      input_tokens_details: { cached_tokens: 96, cache_write_tokens: 112 },
      output_tokens_details: { reasoning_tokens: 6 },
    });
  });

  it("keeps details reported by only one iteration", () => {
    const merged = mergeUsage(
      { input_tokens: 5, output_tokens: 1, total_tokens: 6 },
      {
        input_tokens: 7,
        output_tokens: 1,
        total_tokens: 8,
        input_tokens_details: { cached_tokens: 4, cache_write_tokens: 3 },
      },
    );

    expect(merged?.input_tokens_details).toEqual({
      cached_tokens: 4,
      cache_write_tokens: 3,
    });
  });

  it("omits details entirely when neither iteration reports them", () => {
    const merged = mergeUsage(
      { input_tokens: 5, output_tokens: 1, total_tokens: 6 },
      { input_tokens: 7, output_tokens: 1, total_tokens: 8 },
    );

    expect(merged).not.toHaveProperty("input_tokens_details");
    expect(merged).not.toHaveProperty("output_tokens_details");
  });
});

describe("responses pass-through request building", () => {
  const passthroughBackend = (
    captured: CreateResponseRequest[],
  ): BackendAdapter => ({
    name: "passthrough-capture",
    mode: "responses",
    respond: async (req) => {
      // Simulate the HTTP boundary: only JSON-serializable fields survive.
      captured.push(JSON.parse(JSON.stringify(req)) as CreateResponseRequest);
      return {
        id: "resp_1",
        object: "response",
        created_at: 1,
        status: "completed",
        error: null,
        incomplete_details: null,
        instructions: null,
        max_output_tokens: null,
        model: req.model,
        output: [],
        parallel_tool_calls: true,
        previous_response_id: null,
        conversation: null,
        temperature: null,
        tool_choice: "auto",
        tools: [],
        top_p: null,
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        user: null,
        metadata: null,
      };
    },
  });

  it("strips signal/conversation/maxIterations, forwards store, keeps service_tier + reasoning", async () => {
    const captured: CreateResponseRequest[] = [];
    const agent = new AgentLoop({ backend: passthroughBackend(captured) });

    const ctrl = new AbortController();
    await agent.run({
      request: {
        model: "test-model",
        input: "hi",
        conversation: "conv_123",
        store: true,
        service_tier: "priority",
        reasoning: { effort: "medium", summary: "auto" },
        signal: ctrl.signal,
        maxIterations: 4,
      } as CreateResponseRequest & { signal?: AbortSignal },
      history: [],
      signal: ctrl.signal,
    });

    expect(captured).toHaveLength(1);
    const body = captured[0] as unknown as Record<string, unknown>;
    expect(body).not.toHaveProperty("signal");
    expect(body).not.toHaveProperty("maxIterations");
    expect(body).not.toHaveProperty("conversation");
    expect(body.store).toBe(true);
    expect(body).not.toHaveProperty("previous_response_id");
    expect(body.service_tier).toBe("priority");
    expect(body.reasoning).toEqual({ effort: "medium", summary: "auto" });
  });
});

describe("served service_tier propagation", () => {
  it("surfaces the tier the backend actually served on run()", async () => {
    const agent = new AgentLoop({
      backend: {
        name: "tiered",
        mode: "completions",
        complete: async () => ({ ...completion(), service_tier: "default" }),
      },
    });

    const result = await agent.run({
      request: { model: "test-model", input: "hi", service_tier: "priority" },
      history: [],
    });

    expect(result.serviceTier).toBe("default");
  });

  it("leaves serviceTier null when the backend reports none", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: captureBackend(captured) });

    const result = await agent.run({
      request: { model: "test-model", input: "hi", service_tier: "priority" },
      history: [],
    });

    expect(result.serviceTier).toBeNull();
  });

  it("surfaces the served tier from stream chunks", async () => {
    const chunk = (
      delta: ChatCompletionChunk["choices"][0]["delta"],
      finish: ChatCompletionChunk["choices"][0]["finish_reason"] = null,
      extra: Partial<ChatCompletionChunk> = {},
    ): ChatCompletionChunk => ({
      id: "chatcmpl-1",
      object: "chat.completion.chunk",
      created: 1,
      model: "test-model",
      choices: [{ index: 0, delta, finish_reason: finish }],
      ...extra,
    });

    const agent = new AgentLoop({
      backend: {
        name: "tiered-stream",
        mode: "completions",
        stream: async function* () {
          yield chunk({ role: "assistant", content: "he" });
          yield chunk({ content: "y" }, null, { service_tier: "default" });
          yield chunk({}, "stop");
        },
      },
    });

    const gen = agent.stream({
      request: { model: "test-model", input: "hi", service_tier: "priority" },
      history: [],
    });
    let result;
    while (true) {
      const r = await gen.next();
      if (r.done) {
        result = r.value;
        break;
      }
    }

    expect(result.serviceTier).toBe("default");
  });
});

describe("backend resolution", () => {
  const named = (name: string, hits: string[]): BackendAdapter => ({
    name,
    mode: "completions",
    complete: async () => {
      hits.push(name);
      return completion(`from-${name}`);
    },
  });

  it("runs against the backend it was constructed with", async () => {
    const hits: string[] = [];
    const agent = new AgentLoop({ backend: named("constructed", hits) });

    await agent.run({ request: { model: "m", input: "hi" }, history: [] });

    expect(hits).toEqual(["constructed"]);
  });

  it("requires a backend at construction", () => {
    // @ts-expect-error — `backend` is not optional; a loop with no backend to
    // call is not constructible, rather than failing on first use.
    expect(() => new AgentLoop({})).not.toThrow();
  });
});

// ---- maxIterations ----------------------------------------------------------

// The loop opens MCP connections itself, so the connection class is swapped for
// an in-memory fake that records every tool call it executes.
const mcpState = vi.hoisted(() => ({ calls: [] as string[] }));

vi.mock("./mcp/client.js", async (importOriginal) => {
  const mod = await importOriginal<typeof import("./mcp/client.js")>();
  class FakeMcpConnection {
    constructor(public readonly def: unknown) {}
    async listTools() {
      return [
        {
          name: "lookup",
          description: "Looks something up.",
          input_schema: { type: "object", properties: {} },
        },
      ];
    }
    async call(name: string) {
      mcpState.calls.push(name);
      return { output: `result-${mcpState.calls.length}`, isError: false };
    }
    async close() {}
  }
  return { ...mod, McpConnection: FakeMcpConnection };
});

const mcpTool = {
  type: "mcp" as const,
  server_label: "docs",
  server_url: "https://docs.example.test/mcp",
  require_approval: "never" as const,
};

const toolCallCompletion = (n: number): ChatCompletionResponse => ({
  id: `chatcmpl-${n}`,
  object: "chat.completion",
  created: 1,
  model: "test-model",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: `call_${n}`,
            type: "function",
            function: { name: "lookup", arguments: "{}" },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

/** A model that never stops calling the MCP tool — only the cap ends the turn. */
const relentlessBackend = (
  captured: ChatCompletionRequest[],
): BackendAdapter => ({
  name: "relentless",
  mode: "completions",
  complete: async (req) => {
    captured.push(req);
    return toolCallCompletion(captured.length);
  },
});

const streamChunk = (
  delta: ChatCompletionChunk["choices"][0]["delta"],
  finish: ChatCompletionChunk["choices"][0]["finish_reason"] = null,
): ChatCompletionChunk => ({
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 1,
  model: "test-model",
  choices: [{ index: 0, delta, finish_reason: finish }],
});

describe("maxIterations", () => {
  beforeEach(() => {
    mcpState.calls.length = 0;
  });

  it("a request-level maxIterations overrides the client-level cap", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({
      backend: relentlessBackend(captured),
      maxIterations: 5,
    });

    const result = await agent.run({
      request: {
        model: "test-model",
        input: "hi",
        tools: [mcpTool],
        maxIterations: 2,
      },
      history: [],
    });

    expect(captured).toHaveLength(2);
    expect(mcpState.calls).toHaveLength(2);
    expect(result.status).toBe("incomplete");
    expect(result.incompleteDetails).toEqual({ reason: "max_tool_calls" });
    expect(result.items.filter((it) => it.type === "mcp_call")).toHaveLength(2);
    expect(result.items.some((it) => it.type === "function_call")).toBe(false);
  });

  it("falls back to the client-level cap when the request sets none", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({
      backend: relentlessBackend(captured),
      maxIterations: 3,
    });

    const result = await agent.run({
      request: { model: "test-model", input: "hi", tools: [mcpTool] },
      history: [],
    });

    expect(captured).toHaveLength(3);
    expect(result.status).toBe("incomplete");
  });

  it("defaults to 30 round-trips when neither the request nor the client sets one", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({ backend: relentlessBackend(captured) });

    const result = await agent.run({
      request: { model: "test-model", input: "hi", tools: [mcpTool] },
      history: [],
    });

    expect(captured).toHaveLength(30);
    expect(result.status).toBe("incomplete");
  });

  it("is a ceiling, not a count — a turn that ends on its own is untouched", async () => {
    const captured: ChatCompletionRequest[] = [];
    const agent = new AgentLoop({
      backend: {
        name: "answers-second-time",
        mode: "completions",
        complete: async (req) => {
          captured.push(req);
          return captured.length === 1
            ? toolCallCompletion(1)
            : completion("done");
        },
      },
    });

    const result = await agent.run({
      request: {
        model: "test-model",
        input: "hi",
        tools: [mcpTool],
        maxIterations: 5,
      },
      history: [],
    });

    expect(captured).toHaveLength(2);
    expect(mcpState.calls).toHaveLength(1);
    expect(result.status).toBeUndefined();
    expect(result.incompleteDetails).toBeUndefined();
  });

  it.each([0, -1, 1.5])(
    "rejects %s as a request-level maxIterations before touching backend or MCP",
    async (bad) => {
      const captured: ChatCompletionRequest[] = [];
      const agent = new AgentLoop({ backend: relentlessBackend(captured) });

      await expect(
        agent.run({
          request: {
            model: "test-model",
            input: "hi",
            tools: [mcpTool],
            maxIterations: bad,
          },
          history: [],
        }),
      ).rejects.toThrow("`maxIterations` must be a positive integer");

      expect(captured).toHaveLength(0);
      expect(mcpState.calls).toHaveLength(0);
    },
  );

  it("honors the request-level cap when streaming", async () => {
    let calls = 0;
    const backend: BackendAdapter = {
      name: "relentless-stream",
      mode: "completions",
      stream: async function* () {
        calls++;
        yield streamChunk({
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: `call_${calls}`,
              type: "function",
              function: { name: "lookup", arguments: "" },
            },
          ],
        });
        yield streamChunk({
          tool_calls: [{ index: 0, function: { arguments: "{}" } }],
        });
        yield streamChunk({}, "tool_calls");
      },
    };
    const agent = new AgentLoop({ backend, maxIterations: 5 });

    const gen = agent.stream({
      request: {
        model: "test-model",
        input: "hi",
        tools: [mcpTool],
        maxIterations: 2,
      },
      history: [],
    });
    const mcpCallsDone: string[] = [];
    let result;
    while (true) {
      const r = await gen.next();
      if (r.done) {
        result = r.value;
        break;
      }
      if (
        r.value.type === "response.output_item.done" &&
        r.value.item.type === "mcp_call"
      ) {
        mcpCallsDone.push(r.value.item.name);
      }
    }

    expect(calls).toBe(2);
    expect(mcpState.calls).toHaveLength(2);
    expect(mcpCallsDone).toEqual(["lookup", "lookup"]);
    expect(result.status).toBe("incomplete");
    expect(result.incompleteDetails).toEqual({ reason: "max_tool_calls" });
  });
});
