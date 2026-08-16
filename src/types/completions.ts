/**
 * Types for the OpenAI-compatible Chat Completions API (the backend we call).
 * Matches https://api.openai.com/v1/chat/completions as well as vLLM,
 * Ollama's /v1/chat/completions, llama.cpp server, LiteLLM, Together, Groq.
 */

export type CompletionsRole =
  | "system"
  | "developer"
  | "user"
  | "assistant"
  | "tool";

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageUrlContentPart {
  type: "image_url";
  image_url: { url: string; detail?: "auto" | "low" | "high" };
}

/**
 * Document attachment part. OpenAI accepts `file_id` or base64 `file_data`;
 * OpenRouter's file-parser plugin additionally accepts a plain URL in
 * `file_data`. Only valid on user messages.
 */
export interface FileContentPart {
  type: "file";
  file: {
    filename?: string;
    file_data?: string;
    file_id?: string;
  };
}

export type CompletionsContentPart =
  | TextContentPart
  | ImageUrlContentPart
  | FileContentPart;

export interface SystemMessage {
  role: "system" | "developer";
  content: string | TextContentPart[];
  name?: string;
}

export interface UserMessage {
  role: "user";
  content: string | CompletionsContentPart[];
  name?: string;
}

export interface AssistantMessage {
  role: "assistant";
  content?: string | CompletionsContentPart[] | null;
  name?: string;
  tool_calls?: ChatToolCall[];
  refusal?: string | null;
  reasoning_content?: string | null;
  reasoning?: string | null;
  reasoning_details?: ReasoningDetail[] | null;
}

export interface ToolMessage {
  role: "tool";
  content: string | TextContentPart[];
  tool_call_id: string;
}

export type ChatMessage =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolMessage;

export interface ChatToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    /** JSON-encoded string in OpenAI spec. Ollama native returns an object. */
    arguments: string;
  };
}

export interface ChatFunctionTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
    strict?: boolean;
  };
}

export type ChatToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  tools?: ChatFunctionTool[];
  tool_choice?: ChatToolChoice;
  parallel_tool_calls?: boolean;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  stop?: string | string[];
  n?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  response_format?:
    | { type: "text" }
    | { type: "json_object" }
    | {
        type: "json_schema";
        json_schema: {
          name: string;
          schema: Record<string, unknown>;
          strict?: boolean;
        };
      };
  seed?: number;
  user?: string;
  metadata?: Record<string, string>;
  reasoning_effort?: "minimal" | "low" | "medium" | "high";
  service_tier?: "auto" | "default" | "flex" | "priority";
  reasoning?: {
    effort?: "minimal" | "low" | "medium" | "high";
    summary?: "auto" | "concise" | "detailed";
  };
  usage?: { include?: boolean };
}

export interface ChatCompletionChoice {
  index: number;
  message: AssistantMessage;
  finish_reason:
    | "stop"
    | "length"
    | "tool_calls"
    | "content_filter"
    | "function_call"
    | null;
  logprobs?: unknown;
}


export interface UsageCostDetails {
  upstream_inference_cost?: number | null;
  upstream_inference_prompt_cost?: number | null;
  upstream_inference_completions_cost?: number | null;
}

export interface PromptTokensDetails {
  cached_tokens?: number;
  /**
   * Prompt tokens written to the cache. Only reported on GPT-5.6 and later,
   * where writes are billed at 1.25x the uncached input rate (earlier models
   * cache for free and omit the field).
   */
  cache_write_tokens?: number;
  [k: string]: number | undefined;
}

export interface CompletionTokensDetails {
  reasoning_tokens?: number;
  [k: string]: number | undefined;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: PromptTokensDetails;
  completion_tokens_details?: CompletionTokensDetails;
  cost?: number;
  cost_details?: UsageCostDetails;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: ChatCompletionUsage;
  // The tier that actually processed the request — may differ from the
  // requested one (OpenAI downgrades priority past the ramp-rate limit).
  service_tier?: string | null;
  system_fingerprint?: string;
}

// ---- Streaming -----------------------------------------------------------

export interface ChatDeltaToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

export interface ReasoningDetail {
  type: "reasoning.text" | "reasoning.summary" | "reasoning.encrypted" | string;
  text?: string;
  summary?: string;
  data?: string;
  id?: string;
  format?: string;
  index?: number;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: CompletionsRole;
    content?: string | null;
    tool_calls?: ChatDeltaToolCall[];
    refusal?: string | null;
    reasoning_content?: string | null;
    reasoning?: string | null;
    reasoning_details?: ReasoningDetail[] | null;
  };
  finish_reason: ChatCompletionChoice["finish_reason"];
  logprobs?: unknown;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: ChatCompletionUsage;
  // See ChatCompletionResponse.service_tier — the served tier.
  service_tier?: string | null;
  system_fingerprint?: string;
}
