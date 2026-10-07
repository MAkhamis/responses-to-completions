import type { ConversationObject, ResponseObject } from "../types/responses.js";
import {
  attemptSignal,
  describeErrorBody,
  discardBody,
  isRetryableFetchError,
  isRetryableResponse,
  retryDelayMs,
  sleep,
} from "../util/http.js";
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
  /**
   * How many times a request that failed transiently is sent again. Default
   * 2, so up to three attempts; `0` disables retries.
   *
   * Transient means a dropped connection, a timeout (see `timeoutMs`),
   * 408/409/429 or any 5xx — including the HTML error pages Cloudflare serves
   * in front of api.openai.com. Every read, delete, metadata update and
   * conversation creation is retried on those; a retried creation can at
   * worst leave an unused conversation behind. Item appends are retried only
   * when the failure proves the items did not land — a 429, a connection
   * that never opened, or the server's own `x-should-retry: true` — because
   * a 5xx from a gateway does not say whether the write went through, and a
   * duplicated append corrupts the transcript.
   *
   * The wait before each retry follows the server's `retry-after` hint when
   * it gives one, otherwise 0.5 s doubling to 8 s. Aborting the request's
   * signal stops the retries at once.
   */
  maxRetries?: number;
  /**
   * How long, in ms, a request that is safe to retry may wait for OpenAI to
   * start answering before the attempt is abandoned and retried. Default
   * 60000; `0` disables it. Only the wait for the response headers counts, so
   * a large page that is already downloading is never cut off. Item appends
   * are exempt: an abandoned append might still land.
   */
  timeoutMs?: number;
}

const MAX_ITEMS_PER_CALL = 20;
const MAX_LIST_PAGES = 1000;
const DEFAULT_MAX_RETRIES = 2;
const DEFAULT_TIMEOUT_MS = 60_000;
/** The longest delay `setTimeout` honors — a larger one fires at once. */
const MAX_TIMEOUT_MS = 2_147_483_647;

const ITEM_INCLUDES = ["message.input_image.image_url"] as const;

function withIncludes(qs: URLSearchParams): URLSearchParams {
  for (const inc of ITEM_INCLUDES) qs.append("include[]", inc);
  return qs;
}

/**
 * The retry settings a store will run with, defaults applied. Throws on a
 * value it cannot honor, so a bad config fails when the client is built
 * rather than on its first request.
 */
export function resolveOpenAIStoreRetries(opts: {
  maxRetries?: number;
  timeoutMs?: number;
}): { maxRetries: number; timeoutMs: number } {
  const maxRetries = opts.maxRetries ?? DEFAULT_MAX_RETRIES;
  if (!Number.isInteger(maxRetries) || maxRetries < 0) {
    throw new Error(
      `OpenAIConversationStore: \`maxRetries\` must be a non-negative integer, got ${String(opts.maxRetries)}`,
    );
  }
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (
    !Number.isFinite(timeoutMs) ||
    timeoutMs < 0 ||
    timeoutMs > MAX_TIMEOUT_MS
  ) {
    throw new Error(
      `OpenAIConversationStore: \`timeoutMs\` must be between 0 and ${MAX_TIMEOUT_MS} ms, got ${String(opts.timeoutMs)}`,
    );
  }
  return { maxRetries, timeoutMs };
}

/** Which request failed, for the error message. */
interface RequestContext {
  method?: string;
  /** Path without the query string. */
  path?: string;
  /** Attempts made, counting the first. */
  attempts?: number;
}

function describeRequest(ctx: RequestContext): string {
  const parts: string[] = [];
  const target = [ctx.method, ctx.path].filter(Boolean).join(" ");
  if (target) parts.push(target);
  if (ctx.attempts && ctx.attempts > 1) parts.push(`${ctx.attempts} attempts`);
  return parts.length ? ` [${parts.join(", ")}]` : "";
}

/**
 * A non-2xx answer from the Conversations API. `body` holds the raw response
 * text; the message carries a one-line reading of it — an HTML error page
 * from a gateway is reduced to its title — plus the request and, when
 * retries ran out, how many attempts were made.
 */
export class OpenAIStoreError extends Error {
  /** Attempts made before giving up, counting the first. */
  readonly attempts: number;

  constructor(
    public status: number,
    public body: string,
    context: RequestContext = {},
  ) {
    super(
      `OpenAI conversations API error ${status}: ${describeErrorBody(body)}${describeRequest(context)}`,
    );
    this.name = "OpenAIStoreError";
    this.attempts = context.attempts ?? 1;
  }
}

/**
 * A request OpenAI did not start answering within `timeoutMs`, on every
 * attempt. There is no status or body to report, so it is not an
 * `OpenAIStoreError` — nor an `AbortError`, since the caller did not end it.
 */
export class OpenAIStoreTimeoutError extends Error {
  /** Attempts made before giving up, counting the first. */
  readonly attempts: number;

  constructor(
    public timeoutMs: number,
    context: RequestContext = {},
  ) {
    super(
      `OpenAI conversations API did not respond within ${timeoutMs} ms${describeRequest(context)}`,
    );
    this.name = "OpenAIStoreTimeoutError";
    this.attempts = context.attempts ?? 1;
  }
}

interface CallOptions {
  body?: unknown;
  /** Treat 404 as "absent" and return null rather than throwing. */
  nullOn404?: boolean;
  signal?: AbortSignal;
  /**
   * Whether the request may be sent again after a failure that might still
   * have been applied. Defaults to true for GET and DELETE, false for POST.
   */
  repeatable?: boolean;
}

type AttemptOutcome<T> =
  { done: true; value: T | null } | { done: false; retryInMs: number };

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
  private maxRetries: number;
  private timeoutMs: number;

  constructor(private opts: OpenAIConversationStoreOptions) {
    this.baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(
      /\/+$/,
      "",
    );
    this.fetch = opts.fetch ?? fetch;
    this.pageSize = opts.pageSize ?? 100;
    ({ maxRetries: this.maxRetries, timeoutMs: this.timeoutMs } =
      resolveOpenAIStoreRetries(opts));
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

  /**
   * One Conversations API request, retried on transient failures (see
   * `maxRetries`). A single upstream blip — a 504 page from the gateway in
   * front of api.openai.com — used to fail the whole turn that needed it.
   */
  private async call<T>(
    method: "GET" | "POST" | "DELETE",
    path: string,
    opts: CallOptions = {},
  ): Promise<T | null> {
    for (let attempt = 1; ; attempt++) {
      const outcome = await this.attempt<T>(method, path, opts, attempt);
      if (outcome.done) return outcome.value;
      await sleep(outcome.retryInMs, opts.signal);
    }
  }

  private async attempt<T>(
    method: "GET" | "POST" | "DELETE",
    path: string,
    opts: CallOptions,
    attempt: number,
  ): Promise<AttemptOutcome<T>> {
    const { body, nullOn404 = false, signal } = opts;
    const repeatable = opts.repeatable ?? method !== "POST";
    const canRetry = attempt <= this.maxRetries;
    const context: RequestContext = {
      method,
      path: path.split("?")[0],
      attempts: attempt,
    };
    // Only a request that may be sent again is abandoned on a timeout —
    // giving up on an append would leave unknown whether it landed.
    const scope =
      repeatable && this.timeoutMs > 0
        ? attemptSignal(signal, this.timeoutMs)
        : undefined;
    try {
      let res: Response;
      try {
        res = await this.fetch(`${this.baseUrl}${path}`, {
          method,
          headers: this.headers(),
          ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
          signal: scope?.signal ?? signal,
        });
      } catch (err) {
        const timedOut = scope?.timedOut() === true && !signal?.aborted;
        if (
          canRetry &&
          (timedOut || isRetryableFetchError(err, signal, repeatable))
        ) {
          return { done: false, retryInMs: retryDelayMs(attempt - 1) };
        }
        if (timedOut)
          throw new OpenAIStoreTimeoutError(this.timeoutMs, context);
        throw err;
      }
      scope?.settle();
      if (!res.ok) {
        if (nullOn404 && res.status === 404) return { done: true, value: null };
        if (canRetry && isRetryableResponse(res, repeatable)) {
          await discardBody(res);
          return { done: false, retryInMs: retryDelayMs(attempt - 1, res) };
        }
        throw new OpenAIStoreError(res.status, await res.text(), context);
      }
      if (res.status === 204) return { done: true, value: {} as T };
      const text = await res.text();
      if (!text.trim()) return { done: true, value: {} as T };
      return { done: true, value: JSON.parse(text) as T };
    } finally {
      scope?.release();
    }
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
        body: {
          ...(input.metadata ? { metadata: input.metadata } : {}),
          ...(first.length ? { items: first.map(stripServerFields) } : {}),
        },
        signal,
        // A retry after an ambiguous failure can at worst leave an unused
        // conversation behind; the id returned is always the one to use.
        repeatable: true,
      },
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
      { nullOn404: true, signal },
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
      { body: { metadata: patch.metadata }, nullOn404: true, repeatable: true },
    );
  }

  async deleteConversation(
    id: string,
  ): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<{ id: string; deleted?: boolean }>(
      "DELETE",
      `/conversations/${encodeURIComponent(id)}`,
      { nullOn404: true },
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
          body: {
            items: items
              .slice(i, i + MAX_ITEMS_PER_CALL)
              .map(stripServerFields),
          },
          signal,
          // Items carry no idempotency key: re-sending a batch that had in
          // fact landed would duplicate it in the transcript.
          repeatable: false,
        },
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
    const qs = withIncludes(
      new URLSearchParams({ order, limit: String(clamped) }),
    );
    if (after) qs.set("after", after);
    const res = await this.call<{
      data?: ConversationItem[];
      has_more?: boolean;
      last_id?: string | null;
    }>("GET", `${base}?${qs.toString()}`, { nullOn404: true, signal });
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
    const qs = withIncludes(new URLSearchParams());
    return this.call<ConversationItem>(
      "GET",
      `/conversations/${encodeURIComponent(conversationId)}/items/${encodeURIComponent(itemId)}?${qs.toString()}`,
      { nullOn404: true },
    );
  }

  async deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<unknown>(
      "DELETE",
      `/conversations/${encodeURIComponent(conversationId)}/items/${encodeURIComponent(itemId)}`,
      { nullOn404: true },
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
      { nullOn404: true, signal },
    );
  }

  async deleteResponse(id: string): Promise<{ id: string; deleted: boolean }> {
    const res = await this.call<{ deleted?: boolean }>(
      "DELETE",
      `/responses/${encodeURIComponent(id)}`,
      { nullOn404: true },
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
