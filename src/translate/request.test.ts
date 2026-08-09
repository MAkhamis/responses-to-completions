import { describe, expect, it } from "vitest";
import { itemsToMessages } from "./request.js";
import type { InputItem } from "../types/responses.js";

describe("itemsToMessages content translation", () => {
  it("flattens text-only content parts to a plain string", () => {
    const input: InputItem[] = [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "hello " },
          { type: "input_text", text: "world" },
        ],
      },
    ];
    const msgs = itemsToMessages([], input, undefined);
    expect(msgs).toEqual([{ role: "user", content: "hello world" }]);
  });

  it("keeps input_image parts on user messages as multimodal content", () => {
    const input: InputItem[] = [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "Extract the invoice." },
          { type: "input_image", image_url: "https://cdn.example/a.jpg" },
        ],
      },
    ];
    const msgs = itemsToMessages([], input, undefined);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("user");
    expect(msgs[0].content).toEqual([
      { type: "text", text: "Extract the invoice." },
      { type: "image_url", image_url: { url: "https://cdn.example/a.jpg" } },
    ]);
  });

  it("forwards image detail when present", () => {
    const input: InputItem[] = [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "https://cdn.example/a.jpg", detail: "high" },
        ],
      },
    ];
    const msgs = itemsToMessages([], input, undefined);
    expect(msgs[0].content).toEqual([
      {
        type: "image_url",
        image_url: { url: "https://cdn.example/a.jpg", detail: "high" },
      },
    ]);
  });

  it("flattens image parts to text on non-user roles", () => {
    const input: InputItem[] = [
      {
        type: "message",
        role: "system",
        content: [
          { type: "input_text", text: "instructions" },
          { type: "input_image", image_url: "https://cdn.example/a.jpg" },
        ],
      },
    ];
    const msgs = itemsToMessages([], input, undefined);
    expect(msgs).toEqual([{ role: "system", content: "instructions" }]);
  });

  it("skips input_image parts without a url", () => {
    const input: InputItem[] = [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "just text" },
          { type: "input_image", file_id: "file_123" },
        ],
      },
    ];
    const msgs = itemsToMessages([], input, undefined);
    expect(msgs).toEqual([{ role: "user", content: "just text" }]);
  });
});

describe("itemsToMessages encrypted reasoning replay", () => {
  const encrypted = JSON.stringify([
    { type: "reasoning.encrypted", data: "opaque-blob" },
  ]);
  const history = (model?: string) => [
    { type: "message" as const, role: "user" as const, content: "hi" },
    {
      type: "reasoning" as const,
      id: "rs_1",
      status: "completed" as const,
      content: [],
      encrypted_content: encrypted,
      ...(model ? { model } : {}),
    },
    {
      type: "message" as const,
      id: "msg_1",
      role: "assistant" as const,
      status: "completed" as const,
      content: [{ type: "output_text" as const, text: "hello", annotations: [] }],
    },
  ];

  it("replays encrypted reasoning when the model matches", () => {
    const msgs = itemsToMessages(
      history("openai/gpt-5.4-mini"),
      undefined,
      undefined,
      "openai/gpt-5.4-mini",
    );
    const assistant = msgs.find((m) => m.role === "assistant") as {
      reasoning_details?: unknown[];
    };
    expect(assistant.reasoning_details).toEqual(JSON.parse(encrypted));
  });

  it("drops encrypted reasoning when the conversation switched models", () => {
    const msgs = itemsToMessages(
      history("openai/gpt-5.4-mini"),
      undefined,
      undefined,
      "google/gemini-3.1-flash-lite",
    );
    const assistant = msgs.find((m) => m.role === "assistant");
    expect(assistant).not.toHaveProperty("reasoning_details");
  });

  it("drops unstamped (legacy) encrypted reasoning", () => {
    const msgs = itemsToMessages(
      history(undefined),
      undefined,
      undefined,
      "openai/gpt-5.4-mini",
    );
    const assistant = msgs.find((m) => m.role === "assistant");
    expect(assistant).not.toHaveProperty("reasoning_details");
  });
});
