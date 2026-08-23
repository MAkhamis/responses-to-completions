import type { S3Client, S3ClientConfig } from "@aws-sdk/client-s3";
import type { ConversationObject, ResponseObject } from "../types/responses.js";
import { LocalFileStore } from "./local-file.js";
import { OpenAIConversationStore } from "./openai.js";
import type { ConversationItem, Store } from "./store.js";

/** Store backends the SDK knows how to build. */
export type StoreClient = "S3" | "openAI" | "local";

/** `store_client: "local"` — one JSON file per artifact under `dir`. */
export interface LocalStoreClientConfig {
  /** Directory the store owns, created on demand. */
  dir: string;
}

/** `store_client: "S3"` — an S3 bucket the SDK writes JSON objects into. */
export interface S3StoreClientConfig {
  bucket: string;
  prefix?: string;
  /** Reuse an existing client instead of constructing one. */
  client?: S3Client;
  /** Credentials/region for the client the SDK constructs. */
  clientConfig?: S3ClientConfig;
}

/**
 * `store_client: "openAI"` — state lives in OpenAI's Conversations API. Every
 * key is optional: the provider config's credentials are reused when absent.
 */
export interface OpenAIStoreClientConfig {
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Page size used when reading a whole conversation. */
  pageSize?: number;
}

/** The config shape a given store client accepts. */
export type StoreConfigForClient<C extends StoreClient> = C extends "S3"
  ? S3StoreClientConfig
  : C extends "local"
    ? LocalStoreClientConfig
    : OpenAIStoreClientConfig;

/** Credentials the provider config already carries, reused by `"openAI"`. */
export interface StoreCredentialFallback {
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
}

/**
 * The store that fits a `store_client`. `"openAI"` is only meaningful for
 * traffic OpenAI actually serves — it sends every conversation read and write
 * to api.openai.com — so the client restricts it to `source: "openAI"`.
 *
 * The S3 store is constructed lazily: `@aws-sdk/client-s3` is a heavy import
 * and a client that never persists to S3 should not pay for loading it.
 */
export function createStoreForClient(
  client: StoreClient,
  config?:
    S3StoreClientConfig | OpenAIStoreClientConfig | LocalStoreClientConfig,
  fallback: StoreCredentialFallback = {},
): Store {
  assertStoreClientConfig(client, config, fallback);

  if (client === "local") {
    const c = config as LocalStoreClientConfig;
    return new LocalFileStore(c.dir);
  }

  if (client === "openAI") {
    const c = (config ?? {}) as OpenAIStoreClientConfig;
    const apiKey = (c.apiKey ?? fallback.apiKey) as string;
    const baseUrl = c.baseUrl ?? fallback.baseUrl;
    const fetchImpl = c.fetch ?? fallback.fetch;
    const headers =
      c.headers !== undefined || fallback.headers !== undefined
        ? { ...fallback.headers, ...c.headers }
        : undefined;
    return new OpenAIConversationStore({
      apiKey,
      ...(baseUrl ? { baseUrl } : {}),
      ...(headers ? { headers } : {}),
      ...(fetchImpl ? { fetch: fetchImpl } : {}),
      ...(c.pageSize !== undefined ? { pageSize: c.pageSize } : {}),
    });
  }

  const c = config as S3StoreClientConfig;
  return new LazyStore(async () => {
    const { S3Store } = await import("./s3.js");
    return new S3Store(c);
  });
}

/**
 * Validates a `store_client` + `store_config` pairing without building
 * anything.
 *
 * Split out so `ResponsesClient` can keep failing fast on a bad store config
 * at construction while deferring the actual `createStore()` call to first use
 * — an override of that hook cannot run safely until the subclass's own fields
 * are initialized. Both paths share these checks rather than restating them.
 */
export function assertStoreClientConfig(
  client: StoreClient,
  config?:
    S3StoreClientConfig | OpenAIStoreClientConfig | LocalStoreClientConfig,
  fallback: StoreCredentialFallback = {},
): void {
  if (client === "local") {
    const c = config as LocalStoreClientConfig | undefined;
    if (!c?.dir) {
      throw new Error(
        'createStoreForClient: store_client "local" needs `store_config.dir`.',
      );
    }
    return;
  }
  if (client === "openAI") {
    const c = (config ?? {}) as OpenAIStoreClientConfig;
    if (!(c.apiKey ?? fallback.apiKey)) {
      throw new Error(
        'createStoreForClient: store_client "openAI" needs an API key — pass `store_config.apiKey` or an `apiKey` in the provider config.',
      );
    }
    return;
  }
  const c = config as S3StoreClientConfig | undefined;
  if (!c?.bucket) {
    throw new Error(
      'createStoreForClient: store_client "S3" needs `store_config.bucket`.',
    );
  }
}

/**
 * Defers constructing the real store until its first call, so a module as
 * large as the AWS SDK is only loaded by clients that actually store to S3.
 * Construction happens once; every later call reuses the same instance.
 */
class LazyStore implements Store {
  readonly readsResponsesThrough: boolean | undefined;
  readonly assignsConversationIds: boolean | undefined;
  private pending: Promise<Store> | undefined;

  constructor(
    private readonly load: () => Promise<Store>,
    flags: Pick<Store, "readsResponsesThrough" | "assignsConversationIds"> = {},
  ) {
    this.readsResponsesThrough = flags.readsResponsesThrough;
    this.assignsConversationIds = flags.assignsConversationIds;
  }

  private inner(): Promise<Store> {
    return (this.pending ??= this.load());
  }

  async createConversation(
    input: {
      id?: string;
      metadata?: Record<string, string> | null;
      items?: ConversationItem[];
    },
    signal?: AbortSignal,
  ): Promise<ConversationObject> {
    return (await this.inner()).createConversation(input, signal);
  }

  async getConversation(
    id: string,
    signal?: AbortSignal,
  ): Promise<ConversationObject | null> {
    return (await this.inner()).getConversation(id, signal);
  }

  async updateConversation(
    id: string,
    patch: { metadata?: Record<string, string> | null },
  ): Promise<ConversationObject | null> {
    return (await this.inner()).updateConversation(id, patch);
  }

  async deleteConversation(
    id: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return (await this.inner()).deleteConversation(id);
  }

  async appendItems(
    conversationId: string,
    items: ConversationItem[],
    signal?: AbortSignal,
  ): Promise<void> {
    return (await this.inner()).appendItems(conversationId, items, signal);
  }

  async listItems(
    conversationId: string,
    opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
    signal?: AbortSignal,
  ): Promise<{ items: ConversationItem[]; hasMore: boolean }> {
    return (await this.inner()).listItems(conversationId, opts, signal);
  }

  async getItem(
    conversationId: string,
    itemId: string,
  ): Promise<ConversationItem | null> {
    return (await this.inner()).getItem(conversationId, itemId);
  }

  async deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return (await this.inner()).deleteItem(conversationId, itemId);
  }

  async saveResponse(
    resp: ResponseObject,
    signal?: AbortSignal,
  ): Promise<void> {
    return (await this.inner()).saveResponse(resp, signal);
  }

  async getResponse(
    id: string,
    signal?: AbortSignal,
  ): Promise<ResponseObject | null> {
    return (await this.inner()).getResponse(id, signal);
  }

  async deleteResponse(id: string): Promise<{ id: string; deleted: boolean }> {
    return (await this.inner()).deleteResponse(id);
  }
}
