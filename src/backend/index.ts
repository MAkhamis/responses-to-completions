export type { BackendAdapter } from "./adapter.js";
export {
  OpenAICompatAdapter,
  BackendError,
  type OpenAICompatAdapterOptions,
} from "./openai-compat.js";
export { OllamaAdapter, type OllamaAdapterOptions } from "./ollama.js";
export {
  OpenRouterAdapter,
  type OpenRouterAdapterOptions,
  type OpenRouterProviderPreferences,
} from "./open-router-compat.js";
export {
  createBackendForSource,
  KNOWN_PROVIDER_SOURCES,
  type KnownProviderSource,
  type ProviderConfig,
  type OpenRouterOnlyConfig,
  type OllamaOnlyConfig,
  type ConfigForSource,
  type OpenAIProviderConfig,
  type OpenRouterProviderConfig,
  type OllamaProviderConfig,
  type OllamaCompatProviderConfig,
  type OllamaNativeProviderConfig,
} from "./from-source.js";
