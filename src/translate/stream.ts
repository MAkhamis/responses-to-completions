import type {
  ChatCompletionChunk,
  ChatCompletionUsage,
  ReasoningDetail,
} from "../types/completions.js";
import type {
  FunctionCallItem,
  OutputItem,
  OutputMessageItem,
  ReasoningItem,
  ResponseObject,
} from "../types/responses.js";
import { genFcId, genMessageId, genReasoningId } from "../util/ids.js";
import { translateUsage } from "./response.js";

/**
 * Any Responses-API streaming event. We model them as tagged unions so the
 * HTTP layer can serialize them as SSE with `event: <type>\ndata: <json>`.
 */
export type StreamEvent =
  | {
      type: "response.created";
      sequence_number: number;
      response: ResponseObject;
    }
  | {
      type: "response.in_progress";
      sequence_number: number;
      response: ResponseObject;
    }
  | {
      type: "response.output_item.added";
      sequence_number: number;
      output_index: number;
      item: OutputItem;
    }
  | {
      type: "response.content_part.added";
      sequence_number: number;
      item_id: string;
      output_index: number;
      content_index: number;
      part: { type: "output_text"; text: string; annotations: unknown[] };
    }
  | {
      type: "response.output_text.delta";
      sequence_number: number;
      item_id: string;
      output_index: number;
      content_index: number;
      delta: string;
    }
  | {
      type: "response.output_text.done";
      sequence_number: number;
      item_id: string;
      output_index: number;
      content_index: number;
      text: string;
    }
  | {
      type: "response.content_part.done";
      sequence_number: number;
      item_id: string;
      output_index: number;
      content_index: number;
      part: { type: "output_text"; text: string; annotations: unknown[] };
    }
  | {
      type: "response.function_call_arguments.delta";
      sequence_number: number;
      item_id: string;
      output_index: number;
      delta: string;
    }
  | {
      type: "response.function_call_arguments.done";
      sequence_number: number;
      item_id: string;
      output_index: number;
      arguments: string;
    }
  | {
      type: "response.reasoning_summary_text.delta";
      sequence_number: number;
      item_id: string;
      output_index: number;
      delta: string;
    }
  | {
      type: "response.reasoning_summary_text.done";
      sequence_number: number;
      item_id: string;
      output_index: number;
      text: string;
    }
  | {
      type: "response.output_item.done";
      sequence_number: number;
      output_index: number;
      item: OutputItem;
    }
  | {
      type: "response.completed";
      sequence_number: number;
      response: ResponseObject;
    }
  | {
      type: "response.incomplete";
      sequence_number: number;
      response: ResponseObject;
    }
  | {
      type: "response.failed";
      sequence_number: number;
      response: ResponseObject;
    }
  | {
      type: "response.error";
      sequence_number: number;
      code: string;
      message: string;
    };

/**
 * Translates a stream of OpenAI-compat chat.completion.chunk events into
 * the Responses-API event sequence, starting from an initial in-progress
 * ResponseObject snapshot. Returns the final resolved items + usage (and
 * the served service_tier when the backend reported one) for the caller to
 * persist and emit response.completed.
 *
 * Lifecycle per choice:
 *   1. First content delta  → output_item.added (message) + content_part.added
 *      then output_text.delta* per chunk, then output_text.done + content_part.done
 *      + output_item.done on finish.
 *   2. First tool_call delta → output_item.added (function_call) then
 *      function_call_arguments.delta* per chunk, then function_call_arguments.done
 *      + output_item.done on finish.
 *
 * Assumes single-choice responses (n=1); multi-choice is intentionally
 * unsupported since the Responses API surfaces only one response stream.
 */
export async function* translateChunkStream(
  chunks: AsyncIterable<ChatCompletionChunk>,
  initialResponse: ResponseObject,
  startSeq = 0,
): AsyncGenerator<
  StreamEvent,
  {
    items: OutputItem[];
    usage: ReturnType<typeof translateUsage>;
    serviceTier: string | null;
  }
> {
  let seq = startSeq;

  let reasoningItem: ReasoningItem | null = null;
  let reasoningOutputIndex = -1;
  let reasoningText = "";
  const encryptedBlobs: ReasoningDetail[] = [];

  // Message-output tracking
  let messageItem: OutputMessageItem | null = null;
  let messageOutputIndex = -1;
  let messageText = "";
  let contentOpen = false;

  // Tool-call tracking: index (from chunk) → state
  type ToolState = {
    outputIndex: number;
    item: FunctionCallItem;
    argsBuf: string;
    doneEmitted: boolean;
  };
  const tools = new Map<number, ToolState>();

  let nextOutputIndex = 0;
  let usage: ChatCompletionUsage | undefined;
  let serviceTier: string | null = null;

  const nextSeq = () => seq++;

  for await (const chunk of chunks) {
    if (chunk.usage) usage = chunk.usage;
    if (chunk.service_tier) serviceTier = chunk.service_tier;
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta ?? {};

    const newEncrypted =
      delta.reasoning_details?.filter(
        (d) => d.type === "reasoning.encrypted",
      ) ?? [];
    const reasoningFromDetails =
      delta.reasoning_details
        ?.filter((d) => d.type !== "reasoning.encrypted")
        .map((d) => d.text || d.summary || "")
        .join("") || null;

    const reasoningDelta =
      delta.reasoning_content || delta.reasoning || reasoningFromDetails;
    const hasReasoningText =
      typeof reasoningDelta === "string" && reasoningDelta.length > 0;

    if (newEncrypted.length > 0 || hasReasoningText) {
      if (!reasoningItem) {
        reasoningItem = {
          type: "reasoning",
          id: genReasoningId(),
          status: "in_progress",
          content: [],
        };
        reasoningOutputIndex = nextOutputIndex++;
        yield {
          type: "response.output_item.added",
          sequence_number: nextSeq(),
          output_index: reasoningOutputIndex,
          item: reasoningItem,
        };
      }
    }

    if (newEncrypted.length > 0) encryptedBlobs.push(...newEncrypted);

    if (hasReasoningText) {
      reasoningText += reasoningDelta!;
      yield {
        type: "response.reasoning_summary_text.delta",
        sequence_number: nextSeq(),
        item_id: reasoningItem!.id,
        output_index: reasoningOutputIndex,
        delta: reasoningDelta!,
      };
    }

    // --- text content ---
    if (typeof delta.content === "string" && delta.content.length > 0) {
      if (!messageItem) {
        messageItem = {
          type: "message",
          id: genMessageId(),
          role: "assistant",
          status: "in_progress",
          content: [],
        };
        messageOutputIndex = nextOutputIndex++;
        yield {
          type: "response.output_item.added",
          sequence_number: nextSeq(),
          output_index: messageOutputIndex,
          item: messageItem,
        };
      }
      if (!contentOpen) {
        yield {
          type: "response.content_part.added",
          sequence_number: nextSeq(),
          item_id: messageItem.id,
          output_index: messageOutputIndex,
          content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        };
        contentOpen = true;
      }
      messageText += delta.content;
      yield {
        type: "response.output_text.delta",
        sequence_number: nextSeq(),
        item_id: messageItem.id,
        output_index: messageOutputIndex,
        content_index: 0,
        delta: delta.content,
      };
    }

    // --- tool calls ---
    if (delta.tool_calls?.length) {
      for (const tc of delta.tool_calls) {
        let state = tools.get(tc.index);
        if (!state) {
          const item: FunctionCallItem = {
            type: "function_call",
            id: genFcId(),
            call_id: tc.id ?? "",
            name: tc.function?.name ?? "",
            arguments: "",
            status: "in_progress",
          };
          const outputIndex = nextOutputIndex++;
          state = { outputIndex, item, argsBuf: "", doneEmitted: false };
          tools.set(tc.index, state);
          yield {
            type: "response.output_item.added",
            sequence_number: nextSeq(),
            output_index: outputIndex,
            item,
          };
        }
        // Later chunks may carry id/name after the first.
        if (tc.id && !state.item.call_id) state.item.call_id = tc.id;
        if (tc.function?.name && !state.item.name)
          state.item.name = tc.function.name;

        const argDelta = tc.function?.arguments;
        if (typeof argDelta === "string" && argDelta.length > 0) {
          state.argsBuf += argDelta;
          yield {
            type: "response.function_call_arguments.delta",
            sequence_number: nextSeq(),
            item_id: state.item.id!,
            output_index: state.outputIndex,
            delta: argDelta,
          };
        }
      }
    }

    // --- finish ---
    if (choice.finish_reason) {
      if (reasoningItem) {
        reasoningItem.status = "completed";
        reasoningItem.content = [
          { type: "reasoning_text", text: reasoningText },
        ];
        if (encryptedBlobs.length > 0) {
          reasoningItem.encrypted_content = JSON.stringify(encryptedBlobs);
          reasoningItem.model = initialResponse.model;
          encryptedBlobs.length = 0;
        }
        yield {
          type: "response.reasoning_summary_text.done",
          sequence_number: nextSeq(),
          item_id: reasoningItem.id,
          output_index: reasoningOutputIndex,
          text: reasoningText,
        };
        yield {
          type: "response.output_item.done",
          sequence_number: nextSeq(),
          output_index: reasoningOutputIndex,
          item: reasoningItem,
        };
      }
      if (messageItem) {
        if (contentOpen) {
          yield {
            type: "response.output_text.done",
            sequence_number: nextSeq(),
            item_id: messageItem.id,
            output_index: messageOutputIndex,
            content_index: 0,
            text: messageText,
          };
          yield {
            type: "response.content_part.done",
            sequence_number: nextSeq(),
            item_id: messageItem.id,
            output_index: messageOutputIndex,
            content_index: 0,
            part: { type: "output_text", text: messageText, annotations: [] },
          };
        }
        messageItem.status = "completed";
        messageItem.content = messageText
          ? [{ type: "output_text", text: messageText }]
          : [];
        yield {
          type: "response.output_item.done",
          sequence_number: nextSeq(),
          output_index: messageOutputIndex,
          item: messageItem,
        };
      }
      for (const state of [...tools.values()].sort(
        (a, b) => a.outputIndex - b.outputIndex,
      )) {
        if (state.doneEmitted) continue;
        state.item.arguments = state.argsBuf;
        state.item.status = "completed";
        yield {
          type: "response.function_call_arguments.done",
          sequence_number: nextSeq(),
          item_id: state.item.id!,
          output_index: state.outputIndex,
          arguments: state.argsBuf,
        };
        yield {
          type: "response.output_item.done",
          sequence_number: nextSeq(),
          output_index: state.outputIndex,
          item: state.item,
        };
        state.doneEmitted = true;
      }
    }
  }

  if (reasoningItem && encryptedBlobs.length > 0) {
    reasoningItem.encrypted_content = JSON.stringify(encryptedBlobs);
    // Pin to the requested model — encrypted payloads only replay there.
    reasoningItem.model = initialResponse.model;
  }

  // Assemble final items in output-index order.
  const allItems: OutputItem[] = [];
  if (reasoningItem) allItems[reasoningOutputIndex] = reasoningItem;
  if (messageItem) allItems[messageOutputIndex] = messageItem;
  for (const state of tools.values()) allItems[state.outputIndex] = state.item;
  const items = allItems.filter((x): x is OutputItem => !!x);

  return { items, usage: translateUsage(usage), serviceTier };
}
