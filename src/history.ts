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
 *   doesn't already exist, so clients may pass their own ids).
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
}): Promise<ResolvedHistory> {
  const { request, store } = args;
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
    const existing = await store.getConversation(convId);
    if (!existing) {
      await store.createConversation({ id: convId });
      return { history: [], conversationId: convId, inputItems };
    }
    const { items } = await store.listItems(convId);
    return { history: items, conversationId: convId, inputItems };
  }

  if (request.previous_response_id) {
    if (!store) {
      throw new Error(
        "A store is required to use `previous_response_id`. Pass `store` when constructing ResponsesClient.",
      );
    }
    const prev = await store.getResponse(request.previous_response_id);
    if (!prev) {
      throw new Error(
        `previous_response_id not found: ${request.previous_response_id}`,
      );
    }
    if (prev.conversation?.id) {
      const { items } = await store.listItems(prev.conversation.id);
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
