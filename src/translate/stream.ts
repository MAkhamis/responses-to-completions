import type {
  ChatCompletionChunk,
  ChatCompletionUsage,
} from "../types/completions.js";
import type {
  FunctionCallItem,
  OutputItem,
  OutputMessageItem,
  ResponseObject,
} from "../types/responses.js";
import { genFcId, genMessageId } from "../util/ids.js";
import { translateUsage } from "./response.js";

/**
 * Any Responses-API streaming event. We model them as tagged unions so the
 * HTTP layer can serialize them as SSE with `event: <type>\ndata: <json>`.
 */
export type StreamEvent =
  | { type: "response.created"; sequence_number: number; response: ResponseObject }
  | { type: "response.in_progress"; sequence_number: number; response: ResponseObject }
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
      type: "response.output_item.done";
      sequence_number: number;
      output_index: number;
      item: OutputItem;
    }
  | { type: "response.completed"; sequence_number: number; response: ResponseObject }
  | { type: "response.failed"; sequence_number: number; response: ResponseObject }
  | {
      type: "response.error";
      sequence_number: number;
      code: string;
      message: string;
    };

/**
 * Translates a stream of OpenAI-compat chat.completion.chunk events into
 * the Responses-API event sequence, starting from an initial in-progress
 * ResponseObject snapshot. Returns the final resolved items + usage for
 * the caller to persist and emit response.completed.
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
): AsyncGenerator<StreamEvent, { items: OutputItem[]; usage: ReturnType<typeof translateUsage> }> {
  let seq = startSeq;

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

  const nextSeq = () => seq++;

  for await (const chunk of chunks) {
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta ?? {};

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
        if (tc.function?.name && !state.item.name) state.item.name = tc.function.name;

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
      for (const state of [...tools.values()].sort((a, b) => a.outputIndex - b.outputIndex)) {
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

  // Assemble final items in output-index order.
  const allItems: OutputItem[] = [];
  if (messageItem) allItems[messageOutputIndex] = messageItem;
  for (const state of tools.values()) allItems[state.outputIndex] = state.item;
  const items = allItems.filter((x): x is OutputItem => !!x);

  return { items, usage: translateUsage(usage) };
}
