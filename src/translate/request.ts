import type {
  ChatFunctionTool,
  ChatMessage,
  ChatToolCall,
  ChatToolChoice,
  CompletionsContentPart,
  ReasoningDetail,
} from "../types/completions.js";
import type {
  CreateResponseRequest,
  FunctionCallItem,
  FunctionCallOutputItem,
  InputItem,
  InputMessageItem,
  OutputItem,
  ReasoningItem,
  ResponseTextFormat,
  ToolChoice,
  ToolDef,
} from "../types/responses.js";
import type { ConversationItem } from "../store/store.js";

/**
 * Flattens Responses-API items (history from store + new input) into a
 * chat-completions messages[] array, in chronological order.
 *
 * Mapping rules:
 *   input message (user/system/dev/assistant) → same role + content
 *   function_call (from assistant)            → assistant message with tool_calls
 *   function_call_output (from user)          → tool message (role="tool")
 *   mcp_call (already-executed server-side)   → (omitted — model already saw the
 *                                                tool result as part of the same
 *                                                response; not re-played)
 *   mcp_list_tools                            → (omitted — the tools array itself
 *                                                carries the definitions)
 *   reasoning                                 → (omitted — backend models don't
 *                                                consume reasoning items the way
 *                                                Responses-native ones do)
 *
 * Consecutive assistant tool_calls are merged into a single assistant message
 * so the chat-completions backend sees the shape it expects.
 */
export function itemsToMessages(
  history: ConversationItem[],
  newInput: string | InputItem[] | undefined,
  instructions: string | undefined,
  model?: string,
): ChatMessage[] {
  const msgs: ChatMessage[] = [];

  if (instructions) {
    msgs.push({ role: "system", content: instructions });
  }

  const all: ConversationItem[] = [...history, ...normalizeInput(newInput)];

  let pendingEncrypted: ReasoningDetail[] | null = null;
  for (const item of all) {
    pendingEncrypted = pushItem(msgs, item, pendingEncrypted, model);
  }
  return msgs;
}

function normalizeInput(input: string | InputItem[] | undefined): InputItem[] {
  if (input === undefined) return [];
  if (typeof input === "string") {
    return [{ type: "message", role: "user", content: input }];
  }
  return input;
}

function pushItem(
  msgs: ChatMessage[],
  item: ConversationItem,
  pendingEncrypted: ReasoningDetail[] | null,
  model?: string,
): ReasoningDetail[] | null {
  const t = (item as { type?: string }).type;

  if (t === "reasoning") {
    const r = item as ReasoningItem;
    if (r.encrypted_content && r.model && model && r.model === model) {
      try {
        return JSON.parse(r.encrypted_content) as ReasoningDetail[];
      } catch {}
    }
    return pendingEncrypted;
  }

  // Message item (input or output).
  if (!t || t === "message") {
    const m = item as InputMessageItem;
    const content = messageContentToChatContent(m.content);
    const role = m.role === "developer" ? "system" : m.role;
    if (role === "tool") return pendingEncrypted; // handled via function_call_output
    // Multimodal (image) parts are only valid on user messages in the
    // chat-completions schema; flatten to text for other roles.
    const safeContent =
      typeof content === "string" || role === "user"
        ? content
        : content
            .map((p) => (p.type === "text" ? p.text : ""))
            .join("");
    const msg: ChatMessage = {
      role: role as "system" | "user" | "assistant",
      content: safeContent,
    } as ChatMessage;
    if (role === "assistant" && pendingEncrypted) {
      (msg as { reasoning_details?: ReasoningDetail[] }).reasoning_details =
        pendingEncrypted;
    }
    msgs.push(msg);
    return null;
  }

  if (t === "function_call") {
    const fc = item as FunctionCallItem;
    // Coalesce into the trailing assistant message if it exists.
    const tail = msgs[msgs.length - 1];
    const toolCall: ChatToolCall = {
      id: fc.call_id,
      type: "function",
      function: { name: fc.name, arguments: fc.arguments ?? "" },
    };
    if (tail && tail.role === "assistant") {
      tail.tool_calls = [...(tail.tool_calls ?? []), toolCall];
      if (tail.content === undefined) tail.content = null;
      if (pendingEncrypted) {
        (tail as { reasoning_details?: ReasoningDetail[] }).reasoning_details =
          pendingEncrypted;
      }
      return null;
    } else {
      const msg: ChatMessage = {
        role: "assistant",
        content: null,
        tool_calls: [toolCall],
      };
      if (pendingEncrypted) {
        (msg as { reasoning_details?: ReasoningDetail[] }).reasoning_details =
          pendingEncrypted;
      }
      msgs.push(msg);
      return null;
    }
  }

  if (t === "function_call_output") {
    const fo = item as FunctionCallOutputItem;
    msgs.push({
      role: "tool",
      tool_call_id: fo.call_id,
      content: fo.output ?? "",
    });
    return pendingEncrypted;
  }

  return pendingEncrypted;
}

/**
 * Text-only content flattens to a plain string; content with `input_image`
 * or `input_file` parts becomes a chat-completions multimodal content array
 * so images (vision models) and documents (file-parsing backends) survive
 * the translation.
 */
function messageContentToChatContent(
  content: InputMessageItem["content"] | OutputItem[],
): string | CompletionsContentPart[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: CompletionsContentPart[] = [];
  let hasNonText = false;
  for (const c of content) {
    if (!c || typeof c !== "object") continue;
    const type = (c as { type?: string }).type;
    if (type === "input_text" || type === "output_text") {
      parts.push({ type: "text", text: (c as { text?: string }).text ?? "" });
    } else if (type === "refusal") {
      parts.push({
        type: "text",
        text: (c as { refusal?: string }).refusal ?? "",
      });
    } else if (type === "input_image") {
      const img = c as { image_url?: string; detail?: "auto" | "low" | "high" };
      if (img.image_url) {
        hasNonText = true;
        parts.push({
          type: "image_url",
          image_url: {
            url: img.image_url,
            ...(img.detail ? { detail: img.detail } : {}),
          },
        });
      }
    } else if (type === "input_file") {
      const f = c as {
        file_id?: string;
        file_url?: string;
        file_data?: string;
        filename?: string;
      };
      // Chat-completions carries documents in `file.file_data` (base64 for
      // OpenAI; OpenRouter's file-parser also accepts a plain URL) or by
      // `file_id`. A part with neither has nothing to send — skip it.
      const fileData = f.file_data ?? f.file_url;
      if (fileData || f.file_id) {
        hasNonText = true;
        parts.push({
          type: "file",
          file: {
            ...(f.filename ? { filename: f.filename } : {}),
            ...(fileData ? { file_data: fileData } : {}),
            ...(f.file_id ? { file_id: f.file_id } : {}),
          },
        });
      }
    }
  }
  if (!hasNonText) {
    return parts.map((p) => (p.type === "text" ? p.text : "")).join("");
  }
  return parts;
}

export function translateTools(
  tools: ToolDef[] | undefined,
): ChatFunctionTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  const out: ChatFunctionTool[] = [];
  for (const t of tools) {
    if (t.type === "function") {
      out.push({
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters,
          ...(t.strict !== undefined ? { strict: t.strict } : {}),
        },
      });
    }
    // MCP tools are expanded separately into function tools by the agent loop
    // after mcp_list_tools is resolved; they don't pass through here.
  }
  return out.length ? out : undefined;
}

export function translateToolChoice(
  tc: ToolChoice | undefined,
): ChatToolChoice | undefined {
  if (tc === undefined) return undefined;
  if (typeof tc === "string") return tc;
  if (tc.type === "function")
    return { type: "function", function: { name: tc.name } };
  // MCP tool choice — the agent loop enforces this when selecting tools.
  return "auto";
}

export function translateResponseFormat(
  text: CreateResponseRequest["text"],
): NonNullable<CreateResponseRequest["text"]> extends {
  format?: ResponseTextFormat;
}
  ? import("../types/completions.js").ChatCompletionRequest["response_format"]
  : undefined {
  const fmt = text?.format;
  if (!fmt || fmt.type === "text") return undefined;
  if (fmt.type === "json_object") return { type: "json_object" } as const;
  if (fmt.type === "json_schema") {
    return {
      type: "json_schema",
      json_schema: {
        name: fmt.name,
        schema: fmt.schema,
        ...(fmt.strict !== undefined ? { strict: fmt.strict } : {}),
      },
    } as const;
  }
  return undefined;
}
