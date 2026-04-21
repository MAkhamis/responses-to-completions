/**
 * Types for the OpenAI-compatible Chat Completions API (the backend we call).
 * Matches https://api.openai.com/v1/chat/completions as well as vLLM,
 * Ollama's /v1/chat/completions, llama.cpp server, LiteLLM, Together, Groq.
 */

export type CompletionsRole = "system" | "developer" | "user" | "assistant" | "tool";

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageUrlContentPart {
  type: "image_url";
  image_url: { url: string; detail?: "auto" | "low" | "high" };
}

export type CompletionsContentPart = TextContentPart | ImageUrlContentPart;

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
}

export interface ToolMessage {
  role: "tool";
  content: string | TextContentPart[];
  tool_call_id: string;
}

export type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

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
    | { type: "json_schema"; json_schema: { name: string; schema: Record<string, unknown>; strict?: boolean } };
  seed?: number;
  user?: string;
  metadata?: Record<string, string>;
}

export interface ChatCompletionChoice {
  index: number;
  message: AssistantMessage;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | "function_call" | null;
  logprobs?: unknown;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: ChatCompletionUsage;
  system_fingerprint?: string;
}

// ---- Streaming -----------------------------------------------------------

export interface ChatDeltaToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: CompletionsRole;
    content?: string | null;
    tool_calls?: ChatDeltaToolCall[];
    refusal?: string | null;
    reasoning_content?: string | null;
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
  system_fingerprint?: string;
}
