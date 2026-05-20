/**
 * OpenAI-compatible embeddings API request/response shapes.
 * https://platform.openai.com/docs/api-reference/embeddings
 */

export interface EmbeddingsRequest {
  /** Embedding model id (e.g. "text-embedding-3-small", "nomic-embed-text"). */
  model: string;
  /**
   * Input to embed. Pass a string for a single embedding, an array of strings
   */
  input: string | string[]
  /**
   * Encoding format for returned vectors. Default `"float"`.
   * `"base64"` is more compact on the wire but requires client-side decoding.
   * Not all providers honor this (e.g. Ollama always returns floats).
   */
  encoding_format?: "float" | "base64";
  /**
   * Truncate the embedding to this many dimensions. Only supported by some
   * models (e.g. text-embedding-3-*); silently ignored by providers that
   * don't support it.
   */
  dimensions?: number;
  /** Optional end-user identifier for abuse monitoring. */
  user?: string;
}

export interface EmbeddingObject {
  object: "embedding";
  index: number;
  /** `number[]` when `encoding_format` is `"float"` (default); base64 string when `"base64"`. */
  embedding: number[] | string;
}

export interface EmbeddingsUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface EmbeddingsResponse {
  object: "list";
  data: EmbeddingObject[];
  model: string;
  usage: EmbeddingsUsage;
}
