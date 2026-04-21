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
