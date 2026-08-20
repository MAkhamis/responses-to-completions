import type { ConversationItem, Store } from "./store/store.js";
import type { CreateResponseRequest, InputItem } from "./types/responses.js";

export interface ResolvedHistory {
  history: ConversationItem[];
  conversationId: string | null;
  inputItems: ConversationItem[];
}

/**
 * Resolves the prior conversation history for a Responses-API request.
 *
 * Priority: explicit `conversation` > `previous_response_id` > none.
 *
 * - `conversation` set ⇒ load items from that conversation (auto-create if it
 *   doesn't already exist, so clients may pass their own ids — except on a
 *   store where the provider assigns them, where an unknown id throws).
 * - `previous_response_id` set ⇒ if the prior response was tied to a
 *   conversation continue that conversation; otherwise hand back the prior
 *   response's outputs as ephemeral history with no persistence.
 * - Otherwise ⇒ empty history.
 *
 * `inputItems` is the normalized form of the new turn's `input` field, suitable
 * for appending to the conversation alongside the agent's produced items.
 *
 * Throws if `previous_response_id` is supplied but missing in the store, or
 * if any branch needs a store and one wasn't provided.
 */
export async function resolveHistory(args: {
  request: CreateResponseRequest;
  store: Store | undefined;
  signal?: AbortSignal;
}): Promise<ResolvedHistory> {
  const { request, store, signal } = args;
  const inputItems = normalizeInputItems(request.input);

  if (request.conversation) {
    if (!store) {
      throw new Error(
        "A store is required to use `conversation`. Pass `store` when constructing ResponsesClient.",
      );
    }
    const convId =
      typeof request.conversation === "string"
        ? request.conversation
        : request.conversation.id;
    const existing = await store.getConversation(convId, signal);
    if (!existing) {
      // Auto-create only works where the caller owns the keyspace. On a
      // provider-keyed store the create would be refused for supplying an id
      // at all, which describes the API rather than what went wrong here.
      if (store.assignsConversationIds) {
        throw new Error(
          `conversation not found: ${convId}. This store does not assign conversation ids — the provider does, so a conversation cannot be created under an id chosen here. Call \`conversations.create()\` and pass the id it returns.`,
        );
      }
      await store.createConversation({ id: convId }, signal);
      return { history: [], conversationId: convId, inputItems };
    }
    const { items } = await store.listItems(convId, undefined, signal);
    return { history: items, conversationId: convId, inputItems };
  }

  if (request.previous_response_id) {
    if (!store) {
      throw new Error(
        "A store is required to use `previous_response_id`. Pass `store` when constructing ResponsesClient.",
      );
    }
    const prev = await store.getResponse(request.previous_response_id, signal);
    if (!prev) {
      throw new Error(
        store.readsResponsesThrough
          ? `previous_response_id not found: ${request.previous_response_id}. This store does not persist responses itself — it reads them back from the provider, so only ids the provider issued resolve. A response synthesized from a chat-completions turn has none: continue with \`conversation\`, reach the provider's native endpoint (\`config.endpoint: "responses"\`), or use a store that persists responses ("local"/"S3").`
          : `previous_response_id not found: ${request.previous_response_id}`,
      );
    }
    if (prev.conversation?.id) {
      const { items } = await store.listItems(
        prev.conversation.id,
        undefined,
        signal,
      );
      return {
        history: items,
        conversationId: prev.conversation.id,
        inputItems,
      };
    }
    return {
      history: prev.output as ConversationItem[],
      conversationId: null,
      inputItems,
    };
  }

  return { history: [], conversationId: null, inputItems };
}

function normalizeInputItems(
  input: string | InputItem[] | undefined,
): ConversationItem[] {
  if (input === undefined) return [];
  if (typeof input === "string") {
    return [
      { type: "message", role: "user", content: input },
    ] as ConversationItem[];
  }
  return input as ConversationItem[];
}
