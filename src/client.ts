import { AgentLoop, type AgentRunResult } from "./agent-loop.js";
import type { BackendAdapter } from "./backend/adapter.js";
import {
  createBackendForSource,
  type KnownProviderSource,
  type OllamaProviderConfig,
  type OpenAIProviderConfig,
  type OpenRouterProviderConfig,
  type ProviderConfig,
} from "./backend/from-source.js";
import { resolveHistory } from "./history.js";
import {
  assertStoreClientConfig,
  createStoreForClient,
  type LocalStoreClientConfig,
  type OpenAIStoreClientConfig,
  type S3StoreClientConfig,
  type StoreClient,
} from "./store/from-source.js";
import type { ConversationItem, Store } from "./store/store.js";
import type { StreamEvent } from "./translate/stream.js";
import type {
  EmbeddingsRequest,
  EmbeddingsResponse,
} from "./types/embeddings.js";
import type {
  ConversationObject,
  CreateResponseRequest,
  OutputItem,
  ResponseObject,
  ResponseStatus,
  Usage,
} from "./types/responses.js";
import { genResponseId, now } from "./util/ids.js";

/** The sources the client knows how to reach. */
export type ClientSource = KnownProviderSource;

/** No persistence: `conversations.*` and `responses.{get,del}` throw. */
interface StoreDisabled {
  store?: false;
  store_client?: never;
  store_config?: never;
}

/** Persist to S3 — available to every source. */
interface StoreToS3 {
  store: true;
  store_client: "S3";
  store_config: S3StoreClientConfig;
}

/** Persist to a local directory (development, tests) — every source. */
interface StoreToLocal {
  store: true;
  store_client: "local";
  store_config: LocalStoreClientConfig;
}

/**
 * Persist to OpenAI's Conversations API. Restricted to `source: "openAI"`
 * without a custom `baseUrl`: every read and write ships the transcript to
 * api.openai.com, so pointing another provider's traffic at it would hand
 * OpenAI their conversations — and a custom `baseUrl` names a compat server
 * that has no Conversations API to serve the store.
 * `store_config` may be omitted — the provider config's credentials are used.
 */
interface StoreToOpenAI {
  store: true;
  store_client: "openAI";
  store_config?: OpenAIStoreClientConfig;
}

/**
 * No source: a store-only client. `conversations.*` and `responses.{get,del}`
 * work; `responses.create` and `embeddings.create` throw, having nothing to
 * call. Only meaningful with a store, which the union enforces.
 */
interface SourceNone {
  source?: never;
  config?: never;
}

interface ClientCommonOptions {
  /** Hard cap on backend round-trips per `responses.create` call. Default 10. */
  maxIterations?: number;
}

/**
 * How a client is configured: name a `source`, give it the `config` that
 * source accepts, and say whether (and where) state is persisted. The client
 * builds the adapter and the store itself — no adapter or store classes to
 * import.
 *
 * Every field is typed by the values around it. `config` is the options of
 * the adapter that serves the source — required whenever a `source` is named,
 * and rejected when none is — so OpenRouter's routing keys are refused on
 * `openAI`/`ollama`. `store_client` is only accepted with `store: true`, and
 * `"openAI"` only when the source is `openAI`. `store_config` follows
 * `store_client`.
 *
 * ```ts
 * const client = new ResponsesClient({
 *   source: "openAI",
 *   config: { apiKey: process.env.OPENAI_API_KEY },
 *   store: true,
 *   store_client: "openAI",
 * });
 * ```
 *
 * `source` may be omitted for a store-only client — conversation and response
 * CRUD with no provider credentials in play. Such a client has nothing to call
 * upstream, so the union requires a real store on that branch: a client with
 * neither a source nor a store could do nothing at all.
 *
 * ```ts
 * const reader = new ResponsesClient({
 *   store: true,
 *   store_client: "S3",
 *   store_config: { bucket: "my-bucket" },
 * });
 * ```
 *
 * One client serves one source. To reach two providers, build two clients.
 */
export type ResponsesClientOptions =
  | (ClientCommonOptions & {
      source: "openAI";
      config: OpenAIProviderConfig & { baseUrl?: never };
    } & (StoreDisabled | StoreToS3 | StoreToLocal | StoreToOpenAI))
  | (ClientCommonOptions & {
      source: "openAI";
      config: OpenAIProviderConfig & { baseUrl: string };
    } & (StoreDisabled | StoreToS3 | StoreToLocal))
  | (ClientCommonOptions & {
      source: "openRouter";
      config: OpenRouterProviderConfig;
    } & (StoreDisabled | StoreToS3 | StoreToLocal))
  | (ClientCommonOptions & {
      source: "ollama";
      config: OllamaProviderConfig;
    } & (StoreDisabled | StoreToS3 | StoreToLocal))
  | (ClientCommonOptions & SourceNone & (StoreToS3 | StoreToLocal));

/**
 * High-level SDK client: the OpenAI Responses API surface on top of whichever
 * backend `source` names.
 *
 * ```ts
 * const client = new ResponsesClient({
 *   source: "ollama",
 *   config: { host: "http://localhost:11434" },
 * });
 *
 * const resp = await client.responses.create({ model: "qwen3", input: "hi" });
 * console.log(resp.output_text);
 * ```
 */
export class ResponsesClient {
  /** The source this client serves, or undefined on a store-only client. */
  readonly source: ClientSource | undefined;

  private readonly backendSpec:
    { source: ClientSource; config: ProviderConfig } | undefined;
  private readonly storeSpec:
    | {
        client: StoreClient;
        config?:
          | S3StoreClientConfig
          | LocalStoreClientConfig
          | OpenAIStoreClientConfig;
        providerConfig: ProviderConfig;
      }
    | undefined;
  private readonly maxIterations: number | undefined;
  private backendCache: BackendAdapter | undefined;
  private storeCache: Store | undefined;
  private agentCache: AgentLoop | undefined;

  private get backend(): BackendAdapter | undefined {
    const spec = this.backendSpec;
    if (!spec) return undefined;
    return (this.backendCache ??= this.createBackend(spec.source, spec.config));
  }

  private get store(): Store | undefined {
    const spec = this.storeSpec;
    if (!spec) return undefined;
    return (this.storeCache ??= this.createStore(
      spec.client,
      spec.config,
      spec.providerConfig,
    ));
  }

  private get agent(): AgentLoop | undefined {
    const backend = this.backend;
    if (!backend) return undefined;
    return (this.agentCache ??= new AgentLoop({
      backend,
      maxIterations: this.maxIterations,
    }));
  }

  /**
   * The adapter that fits a source — `openAI` → `OpenAICompatAdapter` with
   * `max_completion_tokens`, `openRouter` → `OpenRouterAdapter` (provider
   * routing, usage accounting), `ollama` → its OpenAI-compatible route, or
   * `OllamaAdapter` on `api: "native"`. Override in a subclass to reach a
   * source differently, or to plug in a custom `BackendAdapter` entirely.
   */
  protected createBackend(
    source: ClientSource,
    config: ProviderConfig,
  ): BackendAdapter {
    return createBackendForSource(source, config);
  }

  /**
   * The store that fits a `store_client`. The source's own credentials are
   * offered as a fallback, so `store_client: "openAI"` needs no second copy
   * of the API key. The provider's `baseUrl` is deliberately not part of the
   * fallback: conversations always go to api.openai.com unless
   * `store_config.baseUrl` says otherwise.
   */
  protected createStore(
    client: StoreClient,
    config:
      | S3StoreClientConfig
      | LocalStoreClientConfig
      | OpenAIStoreClientConfig
      | undefined,
    providerConfig: ProviderConfig,
  ): Store {
    return createStoreForClient(client, config, {
      apiKey: providerConfig.apiKey,
      headers: providerConfig.headers,
      fetch: providerConfig.fetch,
    });
  }

  readonly responses: {
    create: ResponsesCreateOverloads;
    get(id: string): Promise<ResponseObject | null>;
    del(
      id: string,
    ): Promise<{ id: string; object: "response.deleted"; deleted: boolean }>;
  };

  readonly embeddings: {
    create(
      req: EmbeddingsRequest & { signal?: AbortSignal },
    ): Promise<EmbeddingsResponse>;
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

  constructor(options: ResponsesClientOptions) {
    // The public type is a cross-product of source and store unions, which
    // TypeScript won't narrow field-by-field; one internal view of it keeps
    // the checks below readable.
    const opts = options as {
      source?: ClientSource;
      config?: ProviderConfig;
      store?: boolean;
      store_client?: StoreClient;
      store_config?:
        S3StoreClientConfig | LocalStoreClientConfig | OpenAIStoreClientConfig;
      maxIterations?: number;
    };
    const config = opts.config ?? {};

    // A client with neither a source to call nor a store to read is inert;
    // every method on it would throw. The type rejects the combination, so
    // this only catches JS callers.
    if (!opts.source && !opts.store) {
      throw new Error(
        'ResponsesClient: needs a `source` (with its `config`) to reach a provider, a store to read from, or both. Got neither — pass `source: "openAI" | "openRouter" | "ollama"`, or `store: true` with a `store_client`.',
      );
    }

    const needsApiKey =
      opts.source === "openRouter" ||
      (opts.source === "openAI" && !config.baseUrl);
    if (needsApiKey && !config.apiKey) {
      throw new Error(
        `ResponsesClient: source "${opts.source}" requires \`config.apiKey\`.` +
          (opts.source === "openAI"
            ? " A keyless OpenAI-compatible server is reached with a custom `config.baseUrl`, which lifts this requirement."
            : ""),
      );
    }

    this.maxIterations = opts.maxIterations;
    if (opts.source) {
      this.source = opts.source;
      this.backendSpec = { source: opts.source, config };
    }

    if (opts.store) {
      if (!opts.store_client) {
        throw new Error(
          'ResponsesClient: `store: true` requires `store_client` ("S3", "local", or "openAI" when `source` is "openAI").',
        );
      }
      if (opts.store_client === "openAI" && opts.source !== "openAI") {
        throw new Error(
          `ResponsesClient: store_client "openAI" is only available for source "openAI" — it would send ${opts.source} conversations to api.openai.com. Use "S3".`,
        );
      }

      if (opts.store_client === "openAI" && config.baseUrl) {
        throw new Error(
          'ResponsesClient: store_client "openAI" is not available with a custom `config.baseUrl` — that names an OpenAI-compatible server, which has no Conversations API. Use store_client "S3" or "local".',
        );
      }
      // Construction is deferred (see `storeSpec`), but a bad store config is
      // still a construction-time error — deferring that too would surface a
      // typo'd bucket on the first request instead of at startup.
      //
      // Only for the built-in factory, though: these are `createStoreForClient`'s
      // requirements, and a subclass that overrides `createStore` builds
      // something else entirely. Validating unconditionally rejected such a
      // client at construction over a field its override never reads.
      if (this.createStore === ResponsesClient.prototype.createStore) {
        assertStoreClientConfig(opts.store_client, opts.store_config, {
          apiKey: config.apiKey,
          headers: config.headers,
          fetch: config.fetch,
        });
      }
      this.storeSpec = {
        client: opts.store_client,
        config: opts.store_config,
        providerConfig: config,
      };
    }

    const requireStore = (op: string): Store => {
      if (!this.store) {
        throw new Error(
          `ResponsesClient: \`${op}\` requires a store. Construct the client with \`store: true\` and a \`store_client\`.`,
        );
      }
      return this.store;
    };

    const requireBackend = (op: string): BackendAdapter => {
      if (!this.backend) {
        throw new Error(
          `ResponsesClient: \`${op}\` requires a source. This client was built store-only — construct it with a \`source\` and its \`config\` to reach a provider.`,
        );
      }
      return this.backend;
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

    this.embeddings = {
      create: async (req) => {
        const { signal, ...body } = req;
        const backend = requireBackend("embeddings.create");
        if (!backend.embeddings) {
          throw new Error(
            `ResponsesClient: backend "${backend.name}" does not support embeddings.`,
          );
        }
        return backend.embeddings(body, signal);
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
    const agent = this.agent;
    if (!agent) {
      throw new Error(
        "ResponsesClient: `responses.create` requires a source. This client was built store-only — construct it with a `source` and its `config` to reach a provider.",
      );
    }
    if (!req?.model) throw new Error("`model` is required");
    if (req.input === undefined && !req.previous_response_id) {
      throw new Error("`input` or `previous_response_id` is required");
    }

    const store = this.store;
    const { history, conversationId, inputItems } = await resolveHistory({
      request: req,
      store,
      signal: req.signal,
    });

    const responseId = genResponseId();
    const initialResp = buildInitialResponse(responseId, req, conversationId);

    if (req.stream) {
      return new StreamResponse({
        agent,
        store,
        request: req,
        history,
        inputItems,
        conversationId,
        initialResponse: initialResp,
        signal: req.signal,
        persist: req.store !== false,
      });
    }

    let result: AgentRunResult;
    try {
      result = await agent.run({
        request: req,
        history,
        signal: req.signal,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const failedResp: ResponseObject = {
        ...initialResp,
        status: "failed",
        error: { code: "internal_error", message },
      };
      if (req.store !== false && store) {
        await store.saveResponse(failedResp).catch(() => {});
      }
      const e = new Error(message) as Error & { response?: ResponseObject };
      e.response = failedResp;
      throw e;
    }

    const finalResp: ResponseObject = {
      ...initialResp,
      id:
        (req.store !== false ? result.upstreamResponseId : null) ??
        initialResp.id,
      status: result.status ?? "completed",
      ...(result.incompleteDetails
        ? { incomplete_details: result.incompleteDetails }
        : {}),
      output: result.items,
      output_text: aggregateText(result.items),
      usage: result.usage,
      ...(result.serviceTier ? { service_tier: result.serviceTier } : {}),
    };

    if (req.store !== false && store) {
      await persistTurn({
        store,
        conversationId,
        inputItems,
        outputItems: result.items,
        response: finalResp,
      });
    }
    return finalResp;
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
  private finalObserved = false;

  constructor(deps: StreamResponseDeps) {
    this.deps = deps;
    this.finalPromise = new Promise<ResponseObject>((resolve, reject) => {
      this.finalResolve = resolve;
      this.finalReject = reject;
    });
    // This used to be an unconditional `catch(() => {})`, added to silence
    // unhandled-rejection warnings — but the terminal event is emitted before
    // the store write, so a persistence failure reaches only `finalPromise`,
    // and swallowing it left a consumer that just iterates with no signal at
    // all: no event, no log, exit code 0, while the turn was never saved. The
    // emit-before-persist ordering is deliberate and stays; what changes is
    // that a rejection nobody is waiting for is now reported instead of
    // dropped. The deferral gives a caller that awaits `finalResponse()`
    // slightly later than the rejection a chance to claim it first.
    //
    // Only post-delivery failures reach this. A failure before the terminal
    // event is emitted as `response.failed`, which every iterating consumer
    // already sees, so `produce()` marks those observed — otherwise an upstream
    // error or a deliberate `AbortSignal` printed this warning on a correctly
    // handled turn.
    this.finalPromise.catch((err) => {
      setTimeout(() => {
        if (this.finalObserved) return;
        console.error(
          "ResponsesClient: a streamed turn was delivered but its result was never observed, and it failed. Await `stream.finalResponse()` to handle this yourself.",
          err,
        );
      }, 0);
    });
    void this.produce();
  }

  /** Resolves with the persisted final response, or rejects on upstream error. */
  finalResponse(): Promise<ResponseObject> {
    this.finalObserved = true;
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
    let delivered = false;
    let responseId = initialResponse.id;
    let announced = false;
    // `store` is forwarded to a native `/responses` provider, so with
    // `store: false` the provider discards the turn — adopting the id it
    // issued would hand back an id that resolves nowhere.
    const adoptUpstreamId = request.store !== false;

    /**
     * Every event of a stream has to name the same response, and the id isn't
     * settled until a native `/responses` backend reveals the one it issued —
     * so the opening lifecycle pair waits for the first sign of upstream
     * activity instead of going out ahead of it. Nothing observable happens
     * in that window: the consumer pulls from a buffer either way.
     */
    const snapshot = (): ResponseObject => ({
      ...initialResponse,
      id: responseId,
    });
    const announce = (): void => {
      if (announced) return;
      announced = true;
      const response = snapshot();
      this.emit({ type: "response.created", sequence_number: seq++, response });
      this.emit({
        type: "response.in_progress",
        sequence_number: seq++,
        response,
      });
    };

    try {
      let items: OutputItem[] = [];
      let usage: Usage | null = null;
      let servedTier: string | null = null;
      let status: ResponseStatus | undefined;
      let incompleteDetails: { reason: string } | null = null;

      const gen = agent.stream({
        request,
        history,
        signal,
        onUpstreamResponseId: (id) => {
          if (!announced && adoptUpstreamId) responseId = id;
        },
      });
      while (true) {
        const r = await gen.next();
        if (r.done) {
          items = r.value.items;
          usage = r.value.usage;
          servedTier = r.value.serviceTier ?? null;
          status = r.value.status;
          incompleteDetails = r.value.incompleteDetails ?? null;
          if (!announced && adoptUpstreamId && r.value.upstreamResponseId) {
            responseId = r.value.upstreamResponseId;
          }
          break;
        }
        announce();
        this.emit({ ...r.value, sequence_number: seq++ });
      }
      announce();

      const finalResp: ResponseObject = {
        ...snapshot(),
        status: status ?? "completed",
        ...(incompleteDetails ? { incomplete_details: incompleteDetails } : {}),
        output: items,
        output_text: aggregateText(items),
        usage,
        // Report the tier the backend actually served; fall back to the
        // requested tier already on the snapshot when it didn't say.
        ...(servedTier ? { service_tier: servedTier } : {}),
      };
      delivered = true;
      this.emit({
        type:
          status === "incomplete"
            ? "response.incomplete"
            : "response.completed",
        sequence_number: seq++,
        response: finalResp,
      });
      if (persist && store) {
        await persistTurn({
          store,
          conversationId,
          inputItems,
          outputItems: items,
          response: finalResp,
        });
      }
      this.finalResolve(finalResp);
    } catch (err) {
      if (delivered) {
        this.finalReject(err);
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      announce();
      const failedResp: ResponseObject = {
        ...snapshot(),
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
      // The failure is now on the event stream, so it is not an unobserved
      // one — suppress the constructor's warning for it. Without this, an
      // upstream error or a caller's own `abort()` printed a spurious
      // "never observed" error, with a message claiming the turn had been
      // delivered, at every consumer that handles `response.failed`.
      this.finalObserved = true;
      this.finalReject(err);
    } finally {
      this.producerDone = true;
      this.wakeWaiters();
    }
  }
}

// ---- shared helpers ------------------------------------------------------

export class StorePersistenceError extends Error {
  /** The completed response. `status` is `"completed"`, never `"failed"`. */
  readonly response: ResponseObject;

  constructor(response: ResponseObject, cause: unknown) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(
      `Response ${response.id} completed but could not be persisted: ${detail}`,
      { cause },
    );
    this.name = "StorePersistenceError";
    this.response = response;
  }
}

/**
 * Writes a produced turn to the store.
 *
 * Takes no `AbortSignal` by design: the turn has already been produced and
 * billed by the time this runs, and cancelling midway commits the conversation
 * items while dropping the response record, leaving an id that resolves to
 * nothing. Cancellation belongs on the read path and on work that happens
 * before the caller is told the turn completed.
 */
async function persistTurn(args: {
  store: Store;
  conversationId: string | null;
  inputItems: ConversationItem[];
  outputItems: OutputItem[];
  response: ResponseObject;
}): Promise<void> {
  const { store, conversationId, inputItems, outputItems, response } = args;
  try {
    if (conversationId) {
      await store.appendItems(conversationId, [...inputItems, ...outputItems]);
    }
    await store.saveResponse(response);
  } catch (err) {
    throw new StorePersistenceError(response, err);
  }
}

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
    service_tier: body.service_tier ?? null,
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
