import { describe, expect, it } from "vitest";
import { AgentLoop } from "./agent-loop.js";
import type { BackendAdapter } from "./backend/adapter.js";
import type {
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

  it("strips signal/conversation/store and keeps service_tier + reasoning", async () => {
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
    const body = captured[0] as Record<string, unknown>;
    expect(body).not.toHaveProperty("signal");
    expect(body).not.toHaveProperty("conversation");
    expect(body).not.toHaveProperty("store");
    expect(body).not.toHaveProperty("previous_response_id");
    expect(body.service_tier).toBe("priority");
    expect(body.reasoning).toEqual({ effort: "medium", summary: "auto" });
  });
});
