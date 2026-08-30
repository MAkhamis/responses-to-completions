/**
 * responses-to-completions
 *
 * SDK that exposes the OpenAI Responses API as a high-level client class
 * on top of any OpenAI-compatible chat-completions backend. Handles
 * conversation state, MCP tool execution, and streaming event translation.
 */

// Headline export — the SDK client.
export {
  ResponsesClient,
  StreamResponse,
  StorePersistenceError,
  type ResponsesClientOptions,
  type CreateRequestFor,
  type ClientSource,
} from "./client.js";

// Lower-level building blocks for advanced users who want to bypass the client.
export {
  AgentLoop,
  type AgentRunContext,
  type AgentRunResult,
} from "./agent-loop.js";
export { resolveHistory, type ResolvedHistory } from "./history.js";

// Backends
export {
  OpenAICompatAdapter,
  OllamaAdapter,
  OpenRouterAdapter,
  BackendError,
  createBackendForSource,
  KNOWN_PROVIDER_SOURCES,
  type OpenRouterOnlyConfig,
  type OllamaOnlyConfig,
  type ConfigForSource,
  type OpenAIProviderConfig,
  type OpenRouterProviderConfig,
  type OllamaProviderConfig,
  type OllamaCompatProviderConfig,
  type OllamaNativeProviderConfig,
  type BackendAdapter,
  type OpenAICompatAdapterOptions,
  type OllamaAdapterOptions,
  type OpenRouterAdapterOptions,
  type OpenRouterProviderPreferences,
  type KnownProviderSource,
  type ProviderConfig,
} from "./backend/index.js";

// Stores
export {
  LocalFileStore,
  S3Store,
  OpenAIConversationStore,
  OpenAIStoreError,
  createStoreForClient,
  type Store,
  type ConversationItem,
  type S3StoreOptions,
  type OpenAIConversationStoreOptions,
  type StoreClient,
  type StoreConfigForClient,
  type S3StoreClientConfig,
  type LocalStoreClientConfig,
  type OpenAIStoreClientConfig,
} from "./store/index.js";

// MCP
export { McpConnection, needsApproval, type McpToolInfo } from "./mcp/index.js";

// Types
export * from "./types/index.js";

// Translators (exposed for users who want to build their own orchestration)
export {
  itemsToMessages,
  translateTools,
  translateToolChoice,
  translateResponseFormat,
} from "./translate/request.js";
export {
  completionToOutputItems,
  translateUsage,
} from "./translate/response.js";
export { translateChunkStream, type StreamEvent } from "./translate/stream.js";
