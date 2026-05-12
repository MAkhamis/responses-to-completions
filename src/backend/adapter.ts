import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";

/**
 * Backend adapter — abstracts the downstream chat-completions endpoint.
 * Implementations speak HTTP to the model server (vLLM, Ollama, OpenAI, etc.)
 * and return OpenAI-shaped objects regardless of the wire format used.
 */
export interface BackendAdapter {
  readonly name: string;
  complete(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse>;
  stream(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk>;
}
