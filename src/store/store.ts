import type {
  ConversationObject,
  InputItem,
  OutputItem,
  ResponseObject,
} from "../types/responses.js";

/** Any typed item stored in a conversation — union of input and output items. */
export type ConversationItem = InputItem | OutputItem;

/**
 * Pluggable persistence for Responses API state.
 *
 * Two artifacts are persisted:
 *  - Conversations (metadata + an append-only ordered list of items)
 *  - Responses    (the full ResponseObject, retrievable by id)
 *
 * Implementations must be safe for concurrent use across requests in the
 * same process. External concurrency (multiple proxy instances on one S3
 * bucket, say) is not guaranteed and should be handled at the infra layer.
 */
export interface Store {
  /**
   * True when this store does not persist responses itself: `saveResponse` is
   * a no-op and `getResponse` reads through to the provider, so only ids the
   * provider itself issued can be retrieved. Defaults to false — a store that
   * writes responses to its own backing medium leaves it unset. Callers use it
   * to explain an unresolvable `previous_response_id` instead of reporting a
   * bare miss.
   */
  readonly readsResponsesThrough?: boolean;

  /**
   * True when conversation ids belong to the provider rather than to the
   * caller: `createConversation` rejects a supplied `id`, so a conversation
   * cannot be brought into existence under an id chosen here. Defaults to
   * false — a store that owns its own keyspace leaves it unset. Callers use it
   * to explain an unknown `conversation` id instead of attempting an
   * auto-create that cannot succeed.
   */
  readonly assignsConversationIds?: boolean;

  // Conversations
  createConversation(input: {
    id?: string;
    metadata?: Record<string, string> | null;
    items?: ConversationItem[];
  }): Promise<ConversationObject>;
  getConversation(id: string): Promise<ConversationObject | null>;
  updateConversation(
    id: string,
    patch: { metadata?: Record<string, string> | null },
  ): Promise<ConversationObject | null>;
  deleteConversation(id: string): Promise<{ id: string; deleted: boolean }>;

  // Items
  appendItems(conversationId: string, items: ConversationItem[]): Promise<void>;
  listItems(
    conversationId: string,
    opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
  ): Promise<{ items: ConversationItem[]; hasMore: boolean }>;
  getItem(conversationId: string, itemId: string): Promise<ConversationItem | null>;
  deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }>;

  // Responses
  saveResponse(resp: ResponseObject): Promise<void>;
  getResponse(id: string): Promise<ResponseObject | null>;
  deleteResponse(id: string): Promise<{ id: string; deleted: boolean }>;
}
