/**
 * responses-to-completions
 *
 * Drop-in proxy that exposes the OpenAI Responses API on top of any
 * OpenAI-compatible /v1/chat/completions backend.
 */

export { createServer, mountRoutes, type RouteDeps } from "./server/routes.js";
export { SseWriter } from "./server/sse-writer.js";

export { AgentLoop, type AgentRunContext, type AgentRunResult } from "./agent-loop.js";

// Backends
export {
  OpenAICompatAdapter,
  OllamaAdapter,
  BackendError,
  type BackendAdapter,
  type OpenAICompatAdapterOptions,
  type OllamaAdapterOptions,
} from "./backend/index.js";

// Stores
export {
  LocalFileStore,
  S3Store,
  type Store,
  type ConversationItem,
  type S3StoreOptions,
} from "./store/index.js";

// MCP
export { McpConnection, needsApproval, type McpToolInfo } from "./mcp/index.js";

// Types
export * from "./types/index.js";

// Translators (exposed for users who want to build their own server)
export {
  itemsToMessages,
  translateTools,
  translateToolChoice,
  translateResponseFormat,
} from "./translate/request.js";
export { completionToOutputItems, translateUsage } from "./translate/response.js";
export { translateChunkStream, type StreamEvent } from "./translate/stream.js";
