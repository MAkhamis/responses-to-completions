import type { ChatCompletionResponse, ChatCompletionUsage } from "../types/completions.js";
import type {
  FunctionCallItem,
  OutputItem,
  OutputMessageItem,
  Usage,
} from "../types/responses.js";
import { genFcId, genMessageId } from "../util/ids.js";

/**
 * Converts a single chat-completion choice into a list of Responses-API
 * output items. Tool calls become function_call items; assistant content
 * becomes an output message with one output_text content part.
 *
 * Refusals and reasoning traces are propagated when present.
 */
export function completionToOutputItems(
  resp: ChatCompletionResponse,
): { items: OutputItem[]; outputText: string } {
  const choice = resp.choices[0];
  if (!choice) return { items: [], outputText: "" };
  const msg = choice.message;
  const items: OutputItem[] = [];
  let outputText = "";

  // Text / refusal → message item
  const contentText = messageContentToText(msg.content);
  const refusal = msg.refusal ?? null;

  if (contentText || refusal) {
    const message: OutputMessageItem = {
      type: "message",
      id: genMessageId(),
      role: "assistant",
      status: "completed",
      content: [],
    };
    if (refusal) message.content.push({ type: "refusal", refusal });
    if (contentText) {
      message.content.push({ type: "output_text", text: contentText });
      outputText = contentText;
    }
    items.push(message);
  }

  // Tool calls → function_call items
  if (msg.tool_calls?.length) {
    for (const tc of msg.tool_calls) {
      const fc: FunctionCallItem = {
        type: "function_call",
        id: genFcId(),
        call_id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments ?? "",
        status: "completed",
      };
      items.push(fc);
    }
  }

  return { items, outputText };
}

function messageContentToText(c: unknown): string {
  if (typeof c === "string") return c;
  if (!c) return "";
  if (!Array.isArray(c)) return "";
  return c
    .map((p) => {
      if (!p || typeof p !== "object") return "";
      if ((p as { type?: string }).type === "text")
        return (p as { text?: string }).text ?? "";
      return "";
    })
    .join("");
}

export function translateUsage(u?: ChatCompletionUsage): Usage | null {
  if (!u) return null;
  return {
    input_tokens: u.prompt_tokens,
    output_tokens: u.completion_tokens,
    total_tokens: u.total_tokens,
    ...(u.prompt_tokens_details
      ? { input_tokens_details: { cached_tokens: u.prompt_tokens_details.cached_tokens ?? 0 } }
      : {}),
    ...(u.completion_tokens_details
      ? {
          output_tokens_details: {
            reasoning_tokens: u.completion_tokens_details.reasoning_tokens ?? 0,
          },
        }
      : {}),
  };
}
