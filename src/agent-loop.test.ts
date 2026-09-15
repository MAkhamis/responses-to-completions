import { describe, expect, it } from "vitest";
import { AgentLoop, mergeUsage } from "./agent-loop.js";
import type { BackendAdapter } from "./backend/adapter.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "./types/completions.js";
import type {
  CreateResponseRequest,
  InputItem,
} from "./types/responses.js";
import type { ConversationItem } from "./store/store.js";

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

  it("strips signal/conversation, forwards store, keeps service_tier + reasoning", async () => {
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
      } as CreateResponseRequest & { signal?: AbortSignal },
      history: [],
      signal: ctrl.signal,
    });

    expect(captured).toHaveLength(1);
    const body = captured[0] as unknown as Record<string, unknown>;
    expect(body).not.toHaveProperty("signal");
    expect(body).not.toHaveProperty("conversation");
    expect(body.store).toBe(true);
    expect(body).not.toHaveProperty("previous_response_id");
    expect(body.service_tier).toBe("priority");
    expect(body.reasoning).toEqual({ effort: "medium", summary: "auto" });
  });

  it("drops history attachment parts that lost their carrier instead of replaying them", async () => {
    // What OpenAI's Conversations API hands back for a stored image when the
    // read did not ask for `message.input_image.image_url`: the part is still
    // there, but with nothing the provider could fetch. Forwarded as-is it
    // fails the whole turn with "Missing mutually exclusive parameters".
    const strippedImage = {
      type: "input_image",
      detail: "auto",
      file_id: null,
      image_url: null,
    };
    const history: ConversationItem[] = [
      { type: "message", role: "system", content: "You review images." },
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "Review this image" },
          strippedImage,
        ],
      } as unknown as ConversationItem,
      // An image-only message has nothing left once the part goes; an empty
      // content array is rejected too, so the item goes with it.
      {
        type: "message",
        role: "user",
        content: [strippedImage],
      } as unknown as ConversationItem,
      // A file whose data was not returned is the same shape of problem.
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "and this file" },
          { type: "input_file", file_id: null, file_url: null, file_data: null },
        ],
      } as unknown as ConversationItem,
      // Parts that still carry something are replayed untouched.
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "https://x/img.png", detail: "low" },
          { type: "input_image", file_id: "file_123" },
          { type: "input_file", file_data: "JVBERi0=", filename: "a.pdf" },
        ],
      } as unknown as ConversationItem,
      {
        type: "message",
        id: "msg_1",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Looks fine." }],
      },
    ];
    const captured: CreateResponseRequest[] = [];
    const agent = new AgentLoop({ backend: passthroughBackend(captured) });

    await agent.run({
      request: { model: "test-model", input: "thanks" },
      history,
    });

    expect(captured).toHaveLength(1);
    const input = captured[0].input as InputItem[];
    expect(input).toEqual([
      { type: "message", role: "system", content: "You review images." },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Review this image" }],
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "and this file" }],
      },
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "https://x/img.png", detail: "low" },
          { type: "input_image", file_id: "file_123" },
          { type: "input_file", file_data: "JVBERi0=", filename: "a.pdf" },
        ],
      },
      {
        type: "message",
        id: "msg_1",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Looks fine." }],
      },
      { type: "message", role: "user", content: "thanks" },
    ]);
    expect(JSON.stringify(input)).not.toContain('"image_url":null');
  });

  it("forwards a carrier-less part in fresh input untouched, so the provider's own error surfaces", async () => {
    const captured: CreateResponseRequest[] = [];
    const agent = new AgentLoop({ backend: passthroughBackend(captured) });
    const fresh: InputItem[] = [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "look" },
          { type: "input_image" },
        ],
      },
    ];

    await agent.run({
      request: { model: "test-model", input: fresh },
      history: [],
    });

    const input = captured[0].input as InputItem[];
    expect(input).toEqual(fresh);
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
