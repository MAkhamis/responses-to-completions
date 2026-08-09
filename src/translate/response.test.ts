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
