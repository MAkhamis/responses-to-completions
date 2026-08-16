/**
 * Types for the OpenAI Responses API surface this package implements.
 * Spec references: https://platform.openai.com/docs/api-reference/responses
 *
 * Not exhaustive — covers the subset needed to serve clients talking to a
 * chat-completions backend. Built-in tools other than `mcp` are out of scope
 * for v1 (web_search, file_search, code_interpreter, computer_use, image_gen).
 */

import type { UsageCostDetails } from "./completions.js";

export type Role = "system" | "user" | "assistant" | "developer" | "tool";

export interface InputTextContent {
  type: "input_text";
  text: string;
}

export interface InputImageContent {
  type: "input_image";
  image_url?: string;
  file_id?: string;
  detail?: "auto" | "low" | "high";
}

export interface InputFileContent {
  type: "input_file";
  file_id?: string;
  file_url?: string;
  file_data?: string;
  filename?: string;
}

export interface OutputTextContent {
  type: "output_text";
  text: string;
  annotations?: unknown[];
}

export interface RefusalContent {
  type: "refusal";
  refusal: string;
}

export type InputContentPart =
  | InputTextContent
  | InputImageContent
  | InputFileContent;

export type OutputContentPart = OutputTextContent | RefusalContent;

/** Input message item (client-supplied). */
export interface InputMessageItem {
  type?: "message";
  id?: string;
  role: Role;
  content: string | InputContentPart[];
  status?: "in_progress" | "completed" | "incomplete";
}

/** Output message item (assistant-produced). */
export interface OutputMessageItem {
  type: "message";
  id: string;
  role: "assistant";
  status: "in_progress" | "completed" | "incomplete";
  content: OutputContentPart[];
}

/** A function tool call the model wants the client to execute. */
export interface FunctionCallItem {
  type: "function_call";
  id?: string;
  call_id: string;
  name: string;
  /** JSON-encoded string, per OpenAI spec. */
  arguments: string;
  status?: "in_progress" | "completed" | "incomplete";
}

/** Client's response to a function_call. */
export interface FunctionCallOutputItem {
  type: "function_call_output";
  id?: string;
  call_id: string;
  output: string;
  status?: "in_progress" | "completed" | "incomplete";
}

/** Reasoning trace (for reasoning-capable models). */
export interface ReasoningItem {
  type: "reasoning";
  id: string;
  summary?: Array<{ type: "summary_text"; text: string }>;
  content?: Array<{ type: "reasoning_text"; text: string }>;
  encrypted_content?: string;
  status?: "in_progress" | "completed" | "incomplete";
  model?: string;
}

/** Emitted when an MCP server is connected and its tools are listed. */
export interface McpListToolsItem {
  type: "mcp_list_tools";
  id: string;
  server_label: string;
  tools: Array<{
    name: string;
    description?: string;
    input_schema: Record<string, unknown>;
    annotations?: Record<string, unknown>;
  }>;
}

/** Emitted each time the model invokes an MCP tool (resolved server-side). */
export interface McpCallItem {
  type: "mcp_call";
  id: string;
  server_label: string;
  name: string;
  /** JSON-encoded arguments string. */
  arguments: string;
  output: string | null;
  error: string | null;
  approval_request_id: string | null;
}

/** Emitted when a tool call needs user approval. */
export interface McpApprovalRequestItem {
  type: "mcp_approval_request";
  id: string;
  server_label: string;
  name: string;
  arguments: string;
}

/** Client's approval decision for a previous mcp_approval_request. */
export interface McpApprovalResponseItem {
  type: "mcp_approval_response";
  id?: string;
  approval_request_id: string;
  approve: boolean;
  reason?: string;
}

export type InputItem =
  | InputMessageItem
  | FunctionCallItem
  | FunctionCallOutputItem
  | McpApprovalResponseItem
  | ReasoningItem;

export type OutputItem =
  | OutputMessageItem
  | FunctionCallItem
  | ReasoningItem
  | McpListToolsItem
  | McpCallItem
  | McpApprovalRequestItem;

// ---- Tools ----------------------------------------------------------------

export interface FunctionToolDef {
  type: "function";
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
  strict?: boolean;
}

export interface McpToolDef {
  type: "mcp";
  server_label: string;
  server_url?: string;
  connector_id?: string;
  server_description?: string;
  authorization?: string;
  headers?: Record<string, string>;
  allowed_tools?: string[] | { tool_names?: string[] };
  require_approval?: RequireApproval;
  defer_loading?: boolean;
}

export type RequireApproval =
  | "always"
  | "never"
  | { never?: { tool_names?: string[] }; always?: { tool_names?: string[] } };

export type ToolDef = FunctionToolDef | McpToolDef;

export type ToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; name: string }
  | { type: "mcp"; server_label: string; name?: string };

// ---- Request / Response ---------------------------------------------------

export interface CreateResponseRequest {
  model: string;
  input: string | InputItem[];
  instructions?: string;
  previous_response_id?: string;
  conversation?: string | { id: string };
  store?: boolean;
  stream?: boolean;
  tools?: ToolDef[];
  tool_choice?: ToolChoice;
  parallel_tool_calls?: boolean;
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  max_tool_calls?: number;
  metadata?: Record<string, string>;
  reasoning?: { effort?: "minimal" | "low" | "medium" | "high"; summary?: "auto" | "concise" | "detailed" };
  service_tier?: "auto" | "default" | "flex" | "priority";
  text?: { format?: ResponseTextFormat; verbosity?: "low" | "medium" | "high" };
  include?: string[];
  user?: string;
  background?: boolean;
}

export type ResponseTextFormat =
  | { type: "text" }
  | { type: "json_object" }
  | { type: "json_schema"; name: string; schema: Record<string, unknown>; strict?: boolean; description?: string };

export interface InputTokensDetails {
  cached_tokens?: number;
  /**
   * Input tokens written to the cache. Only reported on GPT-5.6 and later,
   * where writes are billed at 1.25x the uncached input rate (earlier models
   * cache for free and omit the field).
   */
  cache_write_tokens?: number;
  [k: string]: number | undefined;
}

export interface OutputTokensDetails {
  reasoning_tokens?: number;
  [k: string]: number | undefined;
}

export interface Usage {
  input_tokens: number;
  input_tokens_details?: InputTokensDetails;
  output_tokens: number;
  output_tokens_details?: OutputTokensDetails;
  total_tokens: number;
  cost?: number;
  cost_details?: UsageCostDetails;
}

export type ResponseStatus =
  | "queued"
  | "in_progress"
  | "completed"
  | "failed"
  | "incomplete"
  | "cancelled";

export interface ResponseObject {
  id: string;
  object: "response";
  created_at: number;
  status: ResponseStatus;
  error: { code: string; message: string } | null;
  incomplete_details: { reason: string } | null;
  instructions: string | null;
  max_output_tokens: number | null;
  model: string;
  output: OutputItem[];
  /** Convenience: aggregated text from all output_text parts. */
  output_text?: string;
  parallel_tool_calls: boolean;
  previous_response_id: string | null;
  conversation: { id: string } | null;
  reasoning?: { effort?: string | null; summary?: string | null };
  service_tier?: string | null;
  temperature: number | null;
  tool_choice: ToolChoice;
  tools: ToolDef[];
  top_p: number | null;
  truncation?: "auto" | "disabled" | null;
  usage: Usage | null;
  user: string | null;
  metadata: Record<string, string> | null;
}

// ---- Conversations --------------------------------------------------------

export interface ConversationObject {
  id: string;
  object: "conversation";
  created_at: number;
  metadata: Record<string, string> | null;
}

export interface ConversationItemsPage {
  object: "list";
  data: OutputItem[] | InputItem[];
  first_id: string | null;
  last_id: string | null;
  has_more: boolean;
}
