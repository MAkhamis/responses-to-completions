import { describe, expect, it } from "vitest";
import { translateUsage } from "./response.js";

describe("translateUsage cost passthrough", () => {
  it("carries cost when the backend reports it", () => {
    const usage = translateUsage({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      cost: 0.00042,
    });

    expect(usage).toEqual({
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      cost: 0.00042,
    });
  });

  it("carries cost_details when the backend reports a breakdown", () => {
    const usage = translateUsage({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      cost: 0.0001,
      cost_details: {
        upstream_inference_prompt_cost: 0.0000485,
        upstream_inference_completions_cost: 0.0000515,
      },
    });

    expect(usage?.cost_details).toEqual({
      upstream_inference_prompt_cost: 0.0000485,
      upstream_inference_completions_cost: 0.0000515,
    });
  });

  it("carries a zero cost (free models)", () => {
    const usage = translateUsage({
      prompt_tokens: 1,
      completion_tokens: 1,
      total_tokens: 2,
      cost: 0,
    });

    expect(usage?.cost).toBe(0);
  });

  it("omits cost when the backend does not report it", () => {
    const usage = translateUsage({
      prompt_tokens: 1,
      completion_tokens: 1,
      total_tokens: 2,
    });

    expect(usage).not.toHaveProperty("cost");
  });
});

describe("translateUsage token details", () => {
  it("carries cache_write_tokens (GPT-5.6+ billable cache writes)", () => {
    const usage = translateUsage({
      prompt_tokens: 2600,
      completion_tokens: 5,
      total_tokens: 2605,
      prompt_tokens_details: { cached_tokens: 2000, cache_write_tokens: 400 },
    });

    expect(usage?.input_tokens_details).toEqual({
      cached_tokens: 2000,
      cache_write_tokens: 400,
    });
  });

  it("preserves detail fields it does not model explicitly", () => {
    const usage = translateUsage({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      prompt_tokens_details: { audio_tokens: 3 },
      completion_tokens_details: { audio_tokens: 2 },
    });

    expect(usage?.input_tokens_details).toEqual({
      audio_tokens: 3,
      cached_tokens: 0,
    });
    expect(usage?.output_tokens_details).toEqual({
      audio_tokens: 2,
      reasoning_tokens: 0,
    });
  });

  it("defaults cached_tokens to 0 when details are reported without it", () => {
    const usage = translateUsage({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      prompt_tokens_details: {},
    });

    expect(usage?.input_tokens_details).toEqual({ cached_tokens: 0 });
  });

  it("drops null detail values instead of leaking them into number fields", () => {
    // OpenRouter-style backends normalize absent detail fields to JSON null.
    const usage = translateUsage({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: null } as never,
      completion_tokens_details: {
        reasoning_tokens: null,
        audio_tokens: 3,
      } as never,
    });

    expect(usage?.input_tokens_details).toEqual({ cached_tokens: 0 });
    expect(usage?.output_tokens_details).toEqual({
      reasoning_tokens: 0,
      audio_tokens: 3,
    });
  });
});
