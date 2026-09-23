import type {
  ChatCompletionResponse,
  ChatCompletionUsage,
  ReasoningDetail,
} from "../types/completions.js";
import type {
  FunctionCallItem,
  OutputItem,
  OutputMessageItem,
  ReasoningItem,
  UrlCitationAnnotation,
  Usage,
} from "../types/responses.js";
import type { ChatUrlCitationAnnotation } from "../types/completions.js";
import { genFcId, genMessageId, genReasoningId } from "../util/ids.js";

/**
 * Converts a single chat-completion choice into a list of Responses-API
 * output items. Tool calls become function_call items; assistant content
 * becomes an output message with one output_text content part.
 *
 * Refusals and reasoning traces are propagated when present.
 */
export function completionToOutputItems(
  resp: ChatCompletionResponse,
  requestModel?: string,
): {
  items: OutputItem[];
  outputText: string;
  /**
   */
  finishReason: string | null;
} {
  const choice = resp.choices[0];
  if (!choice) return { items: [], outputText: "", finishReason: null };
  const msg = choice.message;
  const items: OutputItem[] = [];
  let outputText = "";

  const encryptedBlobs: ReasoningDetail[] =
    msg.reasoning_details?.filter((d) => d.type === "reasoning.encrypted") ??
    [];
  const reasoningFromDetails =
    msg.reasoning_details
      ?.filter((d) => d.type !== "reasoning.encrypted")
      .map((d) => d.text || d.summary || "")
      .join("") || null;

  const reasoningText =
    msg.reasoning_content || msg.reasoning || reasoningFromDetails;
  if (reasoningText || encryptedBlobs.length > 0) {
    const reasoning: ReasoningItem = {
      type: "reasoning",
      id: genReasoningId(),
      status: "completed",
      content: reasoningText
        ? [{ type: "reasoning_text", text: reasoningText }]
        : [],
      ...(encryptedBlobs.length > 0
        ? {
            encrypted_content: JSON.stringify(encryptedBlobs),
            model: requestModel ?? resp.model,
          }
        : {}),
    };
    items.push(reasoning);
  }

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
      const annotations = translateAnnotations(msg.annotations);
      message.content.push({
        type: "output_text",
        text: contentText,
        ...(annotations.length ? { annotations } : {}),
      });
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

  return { items, outputText, finishReason: choice.finish_reason ?? null };
}

/**
 * Chat-completions citations (`{ type, url_citation: {...} }`, OpenRouter's
 * web search) → Responses `url_citation` annotations. Offsets default to 0:
 * OpenRouter does not locate its citations in the text.
 */
export function translateAnnotations(
  annotations: ChatUrlCitationAnnotation[] | null | undefined,
): UrlCitationAnnotation[] {
  const out: UrlCitationAnnotation[] = [];
  for (const a of annotations ?? []) {
    const c =
      a?.type === "url_citation"
        ? (a.url_citation ??
          (a as unknown as ChatUrlCitationAnnotation["url_citation"]))
        : undefined;
    if (!c?.url) continue;
    out.push({
      type: "url_citation",
      url: c.url,
      ...(c.title ? { title: c.title } : {}),
      start_index: typeof c.start_index === "number" ? c.start_index : 0,
      end_index: typeof c.end_index === "number" ? c.end_index : 0,
      ...(c.content ? { content: c.content } : {}),
    });
  }
  return out;
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

function numericDetails(details: object): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(details)) {
    if (typeof v === "number") out[k] = v;
  }
  return out;
}

export function translateUsage(u?: ChatCompletionUsage): Usage | null {
  if (!u) return null;
  return {
    input_tokens: u.prompt_tokens,
    output_tokens: u.completion_tokens,
    total_tokens: u.total_tokens,
    ...(typeof u.cost === "number" ? { cost: u.cost } : {}),
    ...(u.cost_details ? { cost_details: u.cost_details } : {}),
    ...(u.server_tool_use_details
      ? { server_tool_use_details: numericDetails(u.server_tool_use_details) }
      : {}),
    ...(u.prompt_tokens_details
      ? {
          input_tokens_details: {
            ...numericDetails(u.prompt_tokens_details),
            cached_tokens: u.prompt_tokens_details.cached_tokens ?? 0,
          },
        }
      : {}),
    ...(u.completion_tokens_details
      ? {
          output_tokens_details: {
            ...numericDetails(u.completion_tokens_details),
            reasoning_tokens: u.completion_tokens_details.reasoning_tokens ?? 0,
          },
        }
      : {}),
  };
}
