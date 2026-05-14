import { AgentLoop } from "./agent-loop.js";
import type { BackendAdapter } from "./backend/adapter.js";
import { resolveHistory } from "./history.js";
import type { ConversationItem, Store } from "./store/store.js";
import type { StreamEvent } from "./translate/stream.js";
import type {
  ConversationObject,
  CreateResponseRequest,
  OutputItem,
  ResponseObject,
  Usage,
} from "./types/responses.js";
import { genResponseId, now } from "./util/ids.js";

export interface ResponsesClientOptions {
  /**
   * Backend that speaks chat-completions to your model server. Optional —
   * required only for `responses.create`. Conversation CRUD and
   * `responses.{get,del}` work store-only with no backend wired up.
   */
  backend?: BackendAdapter;
  /**
   * Optional store for persisting responses and conversations. If omitted,
   * `conversations.*` and `responses.{get,del}` will throw, and
   * `responses.create` will run without persistence (regardless of `store`).
   */
  store?: Store;
  /** Hard cap on backend round-trips per `responses.create` call. Default 10. */
  maxIterations?: number;
}

/**
 * High-level SDK client. Wraps an {@link AgentLoop} and (optionally) a
 * {@link Store} behind an OpenAI-Responses-style surface.
 *
 * ```ts
 * const client = new ResponsesClient({
 *   backend: new OpenAICompatAdapter({ baseUrl: "...", apiKey: "..." }),
 *   store: new LocalFileStore("./.data"),
 * });
 *
 * const resp = await client.responses.create({ model: "gpt-4o-mini", input: "hi" });
 * console.log(resp.output_text);
 * ```
 */
export class ResponsesClient {
  private readonly backend: BackendAdapter | undefined;
  private readonly store: Store | undefined;
  private readonly agent: AgentLoop | undefined;

  readonly responses: {
    create: ResponsesCreateOverloads;
    get(id: string): Promise<ResponseObject | null>;
    del(
      id: string,
    ): Promise<{ id: string; object: "response.deleted"; deleted: boolean }>;
  };

  readonly conversations: {
    create(input?: {
      id?: string;
      items?: ConversationItem[];
      metadata?: Record<string, string> | null;
    }): Promise<ConversationObject>;
    get(id: string): Promise<ConversationObject | null>;
    update(
      id: string,
      patch: { metadata?: Record<string, string> | null },
    ): Promise<ConversationObject | null>;
    del(id: string): Promise<{
      id: string;
      object: "conversation.deleted";
      deleted: boolean;
    }>;
    items: {
      list(
        conversationId: string,
        opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
      ): Promise<{
        object: "list";
        data: ConversationItem[];
        first_id: string | null;
        last_id: string | null;
        has_more: boolean;
      }>;
      append(
        conversationId: string,
        items: ConversationItem[],
      ): Promise<ConversationItem[]>;
      get(
        conversationId: string,
        itemId: string,
      ): Promise<ConversationItem | null>;
      del(
        conversationId: string,
        itemId: string,
      ): Promise<{
        id: string;
        object: "conversation.item.deleted";
        deleted: boolean;
      }>;
    };
  };

  constructor(opts: ResponsesClientOptions = {}) {
    this.backend = opts.backend;
    this.store = opts.store;
    this.agent = opts.backend
      ? new AgentLoop({
          backend: opts.backend,
          maxIterations: opts.maxIterations,
        })
      : undefined;

    const requireStore = (op: string): Store => {
      if (!this.store) {
        throw new Error(
          `ResponsesClient: \`${op}\` requires a store. Pass \`store\` when constructing the client.`,
        );
      }
      return this.store;
    };

    this.responses = {
      create: ((req: CreateResponseRequest & { signal?: AbortSignal }) =>
        this.createResponse(req)) as ResponsesCreateOverloads,
      get: async (id) => requireStore("responses.get").getResponse(id),
      del: async (id) => {
        const r = await requireStore("responses.del").deleteResponse(id);
        return { id: r.id, object: "response.deleted", deleted: r.deleted };
      },
    };

    this.conversations = {
      create: async (input) =>
        requireStore("conversations.create").createConversation({
          id: input?.id,
          metadata: input?.metadata ?? null,
          items: input?.items ?? [],
        }),
      get: async (id) => requireStore("conversations.get").getConversation(id),
      update: async (id, patch) =>
        requireStore("conversations.update").updateConversation(id, patch),
      del: async (id) => {
        const r =
          await requireStore("conversations.del").deleteConversation(id);
        return {
          id: r.id,
          object: "conversation.deleted",
          deleted: r.deleted,
        };
      },
      items: {
        list: async (conversationId, opts) => {
          const { items, hasMore } = await requireStore(
            "conversations.items.list",
          ).listItems(conversationId, opts);
          return {
            object: "list",
            data: items,
            first_id: items[0] ? (getItemId(items[0]) ?? null) : null,
            last_id: items[items.length - 1]
              ? (getItemId(items[items.length - 1]) ?? null)
              : null,
            has_more: hasMore,
          };
        },
        append: async (conversationId, items) => {
          const s = requireStore("conversations.items.append");
          await s.appendItems(conversationId, items);
          const { items: all } = await s.listItems(conversationId);
          return all;
        },
        get: async (conversationId, itemId) =>
          requireStore("conversations.items.get").getItem(
            conversationId,
            itemId,
          ),
        del: async (conversationId, itemId) => {
          const r = await requireStore("conversations.items.del").deleteItem(
            conversationId,
            itemId,
          );
          return {
            id: r.id,
            object: "conversation.item.deleted",
            deleted: r.deleted,
          };
        },
      },
    };
  }

  // ---- responses.create implementation ----------------------------------
  private async createResponse(
    req: CreateResponseRequest & { signal?: AbortSignal },
  ): Promise<ResponseObject | StreamResponse> {
    if (!this.agent) {
      throw new Error(
        "ResponsesClient: `responses.create` requires a `backend`. Pass `backend` when constructing the client.",
      );
    }
    if (!req?.model) throw new Error("`model` is required");
    if (req.input === undefined && !req.previous_response_id) {
      throw new Error("`input` or `previous_response_id` is required");
    }

    const { history, conversationId, inputItems } = await resolveHistory({
      request: req,
      store: this.store,
    });

    const responseId = genResponseId();
    const initialResp = buildInitialResponse(responseId, req, conversationId);

    if (req.stream) {
      return new StreamResponse({
        agent: this.agent,
        store: this.store,
        request: req,
        history,
        inputItems,
        conversationId,
        initialResponse: initialResp,
        signal: req.signal,
        persist: req.store !== false,
      });
    }

    try {
      const result = await this.agent.run({
        request: req,
        history,
        signal: req.signal,
      });
      const finalResp: ResponseObject = {
        ...initialResp,
        status: "completed",
        output: result.items,
        output_text: aggregateText(result.items),
        usage: result.usage,
      };
      if (req.store !== false && this.store) {
        if (conversationId) {
          await this.store.appendItems(conversationId, [
            ...inputItems,
            ...result.items,
          ]);
        }
        await this.store.saveResponse(finalResp);
      }
      return finalResp;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const failedResp: ResponseObject = {
        ...initialResp,
        status: "failed",
        error: { code: "internal_error", message },
      };
      if (req.store !== false && this.store) {
        await this.store.saveResponse(failedResp).catch(() => {});
      }
      const e = new Error(message) as Error & { response?: ResponseObject };
      e.response = failedResp;
      throw e;
    }
  }
}

// ---- overload typing for client.responses.create -------------------------

interface ResponsesCreateOverloads {
  (
    req: CreateResponseRequest & { stream: true; signal?: AbortSignal },
  ): Promise<StreamResponse>;
  (
    req: CreateResponseRequest & {
      stream?: false | undefined;
      signal?: AbortSignal;
    },
  ): Promise<ResponseObject>;
  (
    req: CreateResponseRequest & { signal?: AbortSignal },
  ): Promise<ResponseObject | StreamResponse>;
}

// ---- streaming wrapper ---------------------------------------------------

interface StreamResponseDeps {
  agent: AgentLoop;
  store: Store | undefined;
  request: CreateResponseRequest;
  history: ConversationItem[];
  inputItems: ConversationItem[];
  conversationId: string | null;
  initialResponse: ResponseObject;
  signal?: AbortSignal;
  persist: boolean;
}

/**
 * Async-iterable result of `responses.create({ stream: true })`.
 *
 * The upstream request is driven eagerly in the background so `finalResponse()`
 * settles whether or not the caller iterates. Events are buffered until a
 * consumer pulls them; a caller that only wants the final response can ignore
 * the iterator entirely.
 */
export class StreamResponse implements AsyncIterable<StreamEvent> {
  private readonly deps: StreamResponseDeps;
  private finalResolve!: (resp: ResponseObject) => void;
  private finalReject!: (err: unknown) => void;
  private readonly finalPromise: Promise<ResponseObject>;
  private iterated = false;
  private readonly queue: StreamEvent[] = [];
  private waiters: Array<() => void> = [];
  private producerDone = false;

  constructor(deps: StreamResponseDeps) {
    this.deps = deps;
    this.finalPromise = new Promise<ResponseObject>((resolve, reject) => {
      this.finalResolve = resolve;
      this.finalReject = reject;
    });
    this.finalPromise.catch(() => {});
    void this.produce();
  }

  /** Resolves with the persisted final response, or rejects on upstream error. */
  finalResponse(): Promise<ResponseObject> {
    return this.finalPromise;
  }

  [Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
    if (this.iterated) {
      throw new Error("StreamResponse can only be iterated once.");
    }
    this.iterated = true;
    return this.iterate();
  }

  private emit(ev: StreamEvent): void {
    this.queue.push(ev);
    this.wakeWaiters();
  }

  private wakeWaiters(): void {
    const ws = this.waiters;
    this.waiters = [];
    for (const w of ws) w();
  }

  private async *iterate(): AsyncGenerator<StreamEvent> {
    while (true) {
      if (this.queue.length > 0) {
        yield this.queue.shift()!;
        continue;
      }
      if (this.producerDone) return;
      await new Promise<void>((resolve) => {
        this.waiters.push(resolve);
      });
    }
  }

  private async produce(): Promise<void> {
    const {
      agent,
      store,
      request,
      history,
      inputItems,
      conversationId,
      initialResponse,
      signal,
      persist,
    } = this.deps;
    let seq = 0;
    let persisted = false;

    try {
      this.emit({
        type: "response.created",
        sequence_number: seq++,
        response: initialResponse,
      });
      this.emit({
        type: "response.in_progress",
        sequence_number: seq++,
        response: initialResponse,
      });

      let items: OutputItem[] = [];
      let usage: Usage | null = null;

      const gen = agent.stream({ request, history, signal });
      while (true) {
        const r = await gen.next();
        if (r.done) {
          items = r.value.items;
          usage = r.value.usage;
          break;
        }
        this.emit({ ...r.value, sequence_number: seq++ });
      }

      const finalResp: ResponseObject = {
        ...initialResponse,
        status: "completed",
        output: items,
        output_text: aggregateText(items),
        usage,
      };
      if (persist && store) {
        if (conversationId) {
          await store.appendItems(conversationId, [...inputItems, ...items]);
        }
        await store.saveResponse(finalResp);
      }
      persisted = true;
      this.emit({
        type: "response.completed",
        sequence_number: seq++,
        response: finalResp,
      });
      this.finalResolve(finalResp);
    } catch (err) {
      if (persisted) {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      const failedResp: ResponseObject = {
        ...initialResponse,
        status: "failed",
        error: { code: "internal_error", message },
      };
      if (persist && store) {
        await store.saveResponse(failedResp).catch(() => {});
      }
      this.emit({
        type: "response.failed",
        sequence_number: seq++,
        response: failedResp,
      });
      this.finalReject(err);
    } finally {
      this.producerDone = true;
      this.wakeWaiters();
    }
  }
}

// ---- shared helpers ------------------------------------------------------

function buildInitialResponse(
  id: string,
  body: CreateResponseRequest,
  conversationId: string | null,
): ResponseObject {
  return {
    id,
    object: "response",
    created_at: now(),
    status: "in_progress",
    error: null,
    incomplete_details: null,
    instructions: body.instructions ?? null,
    max_output_tokens: body.max_output_tokens ?? null,
    model: body.model,
    output: [],
    parallel_tool_calls: body.parallel_tool_calls ?? true,
    previous_response_id: body.previous_response_id ?? null,
    conversation: conversationId ? { id: conversationId } : null,
    temperature: body.temperature ?? null,
    tool_choice: body.tool_choice ?? "auto",
    tools: body.tools ?? [],
    top_p: body.top_p ?? null,
    usage: null,
    user: body.user ?? null,
    metadata: body.metadata ?? null,
  };
}

function aggregateText(items: OutputItem[]): string {
  let out = "";
  for (const it of items) {
    if ((it as { type?: string }).type === "message") {
      const parts =
        (it as { content?: Array<{ type: string; text?: string }> }).content ??
        [];
      for (const p of parts) {
        if (p.type === "output_text" && p.text) out += p.text;
      }
    }
  }
  return out;
}

function getItemId(item: ConversationItem): string | undefined {
  return (item as { id?: string }).id ?? (item as { call_id?: string }).call_id;
}
