import type { ConversationObject, ResponseObject } from "../types/responses.js";
import type { ConversationItem, Store } from "./store.js";

export interface OpenAIConversationStoreOptions {
  apiKey: string;
  /** Base URL without a trailing endpoint. Defaults to the OpenAI API. */
  baseUrl?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /**
   * Page size used when reading a whole conversation. `listItems` with no
   * explicit limit pages until exhaustion, so this only tunes round-trips.
   * Clamped to OpenAI's accepted 1-100 range.
   */
  pageSize?: number;
}

const MAX_ITEMS_PER_CALL = 20;
const MAX_LIST_PAGES = 1000;

export class OpenAIStoreError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`OpenAI conversations API error ${status}: ${body}`);
    this.name = "OpenAIStoreError";
  }
}

/**
 * Store backed by OpenAI's Conversations API instead of local disk or S3 —
 * conversation state lives in the provider's account, and `createConversation`
 * returns the real `conv_…` id OpenAI issued.
 *
 * Use it only for traffic actually served by OpenAI: every read and write
 * sends the conversation's content to api.openai.com, so pointing another
 * provider's requests at this store would hand OpenAI their transcripts.
 *
 * Responses are the one asymmetry. OpenAI persists responses its own
 * `/responses` endpoint created, so `saveResponse` is a no-op here and
 * `getResponse` reads through to the provider. What resolves therefore
 * depends on how the client reaches OpenAI:
 *
 * - `config.endpoint: "responses"` — the turn is served by OpenAI's own
 *   `/responses`, the client adopts the id OpenAI issued for it, and
 *   `previous_response_id` resolves on the read-through.
 * - `config.endpoint: "completions"` (the default) — the `ResponseObject` is
 *   synthesized here from a chat-completions turn and has no counterpart
 *   upstream, so `previous_response_id` cannot resolve. Drive continuity with
 *   `conversation`, which this store persists either way.
 *
 * One consequence of the no-op `saveResponse`: a response served by OpenAI
 * carries no record of the SDK-side conversation it belonged to, so chaining
 * by `previous_response_id` replays that turn's output rather than the whole
 * conversation. Pass `conversation` when the full transcript matters.
 */
export class OpenAIConversationStore implements Store {
  readonly name = "openai-conversations";
  readonly readsResponsesThrough = true;
  readonly assignsConversationIds = true;
  private baseUrl: string;
  private fetch: typeof fetch;
  private pageSize: number;

  constructor(private opts: OpenAIConversationStoreOptions) {
    this.baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(
      /\/+$/,
      "",
    );
    this.fetch = opts.fetch ?? fetch;
    this.pageSize = opts.pageSize ?? 100;
  }

  private headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(this.opts.apiKey
        ? { Authorization: `Bearer ${this.opts.apiKey}` }
        : {}),
      ...this.opts.headers,
    };
  }

  private async call<T>(
    method: "GET" | "POST" | "DELETE",
    path: string,
    body?: unknown,
    /** Treat 404 as "absent" and return null rather than throwing. */
    nullOn404 = false,
    signal?: AbortSignal,
  ): Promise<T | null> {
    const res = await this.fetch(`${this.baseUrl}${path}`, {
      method,
      headers: this.headers(),
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      signal,
    });
    if (!res.ok) {
      if (nullOn404 && res.status === 404) return null;
      throw new OpenAIStoreError(res.status, await res.text());
    }
    if (res.status === 204) return {} as T;
    const text = await res.text();
    if (!text.trim()) return {} as T;
    return JSON.parse(text) as T;
  }

  // ---- conversations ----
  async createConversation(
    input: {
      id?: string;
      metadata?: Record<string, string> | null;
      items?: ConversationItem[];
    },
    signal?: AbortSignal,
  ): Promise<ConversationObject> {
    // OpenAI issues the id; a caller-supplied one cannot be honored, and
    // silently returning a different id would strand the caller's reference.
    if (input.id) {
      throw new Error(
        "OpenAIConversationStore: OpenAI assigns conversation ids — cannot create with a caller-supplied `id`.",
      );
    }
    const items = input.items ?? [];
    const first = items.slice(0, MAX_ITEMS_PER_CALL);
    const created = await this.call<ConversationObject>(
      "POST",
      "/conversations",
      {
        ...(input.metadata ? { metadata: input.metadata } : {}),
        ...(first.length ? { items: first.map(stripServerFields) } : {}),
      },
      false,
      signal,
    );
    if (!created?.id) {
      throw new Error(
        `OpenAIConversationStore: POST ${this.baseUrl}/conversations succeeded but returned no conversation id — cannot address the conversation. The server answered with an empty or unexpected body.`,
      );
    }
    const conversation = created;
    if (items.length > MAX_ITEMS_PER_CALL) {
      await this.appendItems(
        conversation.id,
        items.slice(MAX_ITEMS_PER_CALL),
        signal,
      );
    }
    return conversation;
  }

  async getConversation(
    id: string,
    signal?: AbortSignal,
  ): Promise<ConversationObject | null> {
    return this.call<ConversationObject>(
      "GET",
      `/conversations/${encodeURIComponent(id)}`,
      undefined,
      true,
      signal,
    );
  }

  async updateConversation(
    id: string,
    patch: { metadata?: Record<string, string> | null },
  ): Promise<ConversationObject | null> {
    if (patch.metadata === undefined) return this.getConversation(id);
    return this.call<ConversationObject>(
      "POST",
      `/conversations/${encodeURIComponent(id)}`,
      { metadata: patch.metadata },
      true,
    );
  }

  async deleteConversation(
    id: string,
  ): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<{ id: string; deleted?: boolean }>(
      "DELETE",
      `/conversations/${encodeURIComponent(id)}`,
      undefined,
      true,
    );
    return { id, deleted: res ? (res.deleted ?? true) : false };
  }

  // ---- items ----
  async appendItems(
    conversationId: string,
    items: ConversationItem[],
    signal?: AbortSignal,
  ): Promise<void> {
    for (let i = 0; i < items.length; i += MAX_ITEMS_PER_CALL) {
      await this.call(
        "POST",
        `/conversations/${encodeURIComponent(conversationId)}/items`,
        {
          items: items.slice(i, i + MAX_ITEMS_PER_CALL).map(stripServerFields),
        },
        false,
        signal,
      );
    }
  }

  async listItems(
    conversationId: string,
    opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
    signal?: AbortSignal,
  ): Promise<{ items: ConversationItem[]; hasMore: boolean }> {
    const order = opts?.order ?? "asc";
    const base = `/conversations/${encodeURIComponent(conversationId)}/items`;

    // An explicit limit is one page, verbatim. No limit means "the whole
    // conversation" — the caller is rebuilding history, and stopping at
    // OpenAI's default page size would silently truncate it.
    if (opts?.limit !== undefined) {
      const page = await this.fetchPage(
        base,
        order,
        opts.limit,
        opts.after,
        signal,
      );
      return { items: page.items, hasMore: page.hasMore };
    }

    const all: ConversationItem[] = [];
    // The cursor-advance guard used to return before `all.push(...page.items)`,
    // so a server that returns genuinely new items while echoing the request
    // cursor back in `last_id` lost that whole page. The page is now collected
    // first and the guard runs after, so nothing is dropped — and because it
    // runs unconditionally, a non-advancing cursor still ends the loop on the
    // very next check instead of re-requesting the same page.
    //
    // Dedup by item id additionally guards against a server that repeats items
    // across cursors that *do* advance.
    const seen = new Set<string>();
    let after = opts?.after;
    for (let pages = 0; pages < MAX_LIST_PAGES; pages++) {
      const page = await this.fetchPage(
        base,
        order,
        this.pageSize,
        after,
        signal,
      );
      for (const item of page.items) {
        const id = getItemId(item);
        if (id) {
          if (seen.has(id)) continue;
          seen.add(id);
        }
        all.push(item);
      }
      if (!page.hasMore || !page.lastId) return { items: all, hasMore: false };
      // The cursor did not move, so there is no next page to ask for —
      // requesting it again would return this same page forever. This page's
      // items are already in `all`.
      if (page.lastId === after) return { items: all, hasMore: false };
      after = page.lastId;
    }
    throw new Error(
      `OpenAIConversationStore: conversation ${conversationId} exceeded ${MAX_LIST_PAGES} pages while listing items — aborting rather than paging without bound.`,
    );
  }

  private async fetchPage(
    base: string,
    order: "asc" | "desc",
    limit: number,
    after?: string,
    signal?: AbortSignal,
  ): Promise<{
    items: ConversationItem[];
    hasMore: boolean;
    lastId: string | null;
  }> {
    const clamped = Math.min(100, Math.max(1, Math.trunc(limit)));
    const qs = new URLSearchParams({ order, limit: String(clamped) });
    if (after) qs.set("after", after);
    const res = await this.call<{
      data?: ConversationItem[];
      has_more?: boolean;
      last_id?: string | null;
    }>("GET", `${base}?${qs.toString()}`, undefined, true, signal);
    const items = res?.data ?? [];
    return {
      items,
      hasMore: res?.has_more ?? false,
      lastId:
        res?.last_id ??
        (items.length ? (getItemId(items[items.length - 1]) ?? null) : null),
    };
  }

  async getItem(
    conversationId: string,
    itemId: string,
  ): Promise<ConversationItem | null> {
    return this.call<ConversationItem>(
      "GET",
      `/conversations/${encodeURIComponent(conversationId)}/items/${encodeURIComponent(itemId)}`,
      undefined,
      true,
    );
  }

  async deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<unknown>(
      "DELETE",
      `/conversations/${encodeURIComponent(conversationId)}/items/${encodeURIComponent(itemId)}`,
      undefined,
      true,
    );
    return { id: itemId, deleted: res !== null };
  }

  // ---- responses ----
  /** No-op: OpenAI persists the responses its own endpoint produced. */
  async saveResponse(_resp: ResponseObject): Promise<void> {}

  async getResponse(
    id: string,
    signal?: AbortSignal,
  ): Promise<ResponseObject | null> {
    return this.call<ResponseObject>(
      "GET",
      `/responses/${encodeURIComponent(id)}`,
      undefined,
      true,
      signal,
    );
  }

  async deleteResponse(id: string): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<{ deleted?: boolean }>(
      "DELETE",
      `/responses/${encodeURIComponent(id)}`,
      undefined,
      true,
    );
    return { id, deleted: res ? (res.deleted ?? true) : false };
  }
}

function stripServerFields(item: ConversationItem): ConversationItem {
  const type = (item as { type?: string }).type;

  if (type === "reasoning") {
    const { model: _model, ...rest } = item as ConversationItem & {
      model?: string;
    };
    return rest as ConversationItem;
  }

  const { id: _id, ...rest } = item as ConversationItem & { id?: string };

  if (type === "mcp_list_tools") {
    const { error: _error, ...withoutError } = rest as typeof rest & {
      error?: string;
    };
    return withoutError as ConversationItem;
  }

  return rest as ConversationItem;
}

function getItemId(it: ConversationItem): string | undefined {
  return (it as { id?: string }).id ?? (it as { call_id?: string }).call_id;
}
