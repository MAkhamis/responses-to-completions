import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "../types/completions.js";
import type {
  EmbeddingsRequest,
  EmbeddingsResponse,
} from "../types/embeddings.js";
import type {
  CreateResponseRequest,
  ResponseObject,
} from "../types/responses.js";
import type { StreamEvent } from "../translate/stream.js";

/**
 * Backend adapter — abstracts the downstream model server.
 *
 * Two modes are supported:
 *
 * - `mode === "completions"` — adapter talks to a chat-completions endpoint
 *   (`/v1/chat/completions`-style). Must implement `complete` + `stream`;
 *   the SDK translates Responses-API requests into chat-completions payloads
 *   and back.
 *
 * - `mode === "responses"` — adapter talks to a native Responses endpoint
 *   (`/v1/responses`). Must implement `respond` + `respondStream`; the SDK
 *   forwards Responses-API requests through without translation.
 *
 * A single adapter class may implement both pairs and expose either mode via
 * configuration (see `OpenAICompatAdapter`'s `endpoint` option).
 *
 * Regardless of mode the public SDK surface (`ResponsesClient`) accepts and
 * returns Responses-API shapes.
 */
export interface BackendAdapter {
  readonly name: string;
  readonly mode: "completions" | "responses";

  /** Required when mode === "completions". */
  complete?(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse>;
  stream?(
    req: ChatCompletionRequest,
    signal?: AbortSignal,
  ): AsyncIterable<ChatCompletionChunk>;

  /** Required when mode === "responses". */
  respond?(
    req: CreateResponseRequest,
    signal?: AbortSignal,
  ): Promise<ResponseObject>;
  respondStream?(
    req: CreateResponseRequest,
    signal?: AbortSignal,
  ): AsyncIterable<StreamEvent>;

  embeddings?(
    req: EmbeddingsRequest,
    signal?: AbortSignal,
  ): Promise<EmbeddingsResponse>;
}
