import { describe, expect, it } from "vitest";
import {
  ResponsesClient,
  StorePersistenceError,
  type ResponsesClientOptions,
} from "./client.js";
import type { BackendAdapter } from "./backend/adapter.js";
import type { Store } from "./store/store.js";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";

describe("client construction", () => {
  const capture = (calls: any[], json: any = null) =>
    (async (url: any, init: any) => {
      calls.push({
        url: String(url),
        method: init?.method ?? "GET",
        body: init?.body ? JSON.parse(init.body) : undefined,
        headers: init?.headers,
      });
      const payload = json ?? {
        id: "chatcmpl-1",
        object: "chat.completion",
        created: 1,
        model: "m",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      };
      // One body, read either way — a real Response never disagrees with
      // itself between .json() and .text().
      return {
        ok: true,
        status: 200,
        json: async () => payload,
        text: async () => JSON.stringify(payload),
      };
    }) as unknown as typeof fetch;

  it("builds the openAI adapter from source + config", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: { apiKey: "sk-a", fetch: capture(calls) },
    });

    await client.responses.create({
      model: "m",
      input: "hi",
      max_output_tokens: 32,
      stream: false,
    });

    expect(calls[0].url).toBe("https://api.openai.com/v1/chat/completions");
    expect(calls[0].body).toHaveProperty("max_completion_tokens", 32);
    expect(calls[0].headers.authorization).toBe("Bearer sk-a");
  });

  it("reaches a source by explicit baseUrl", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "ollama",
      config: {
        baseUrl: "http://ollama.internal/v1",
        apiKey: "k",
        fetch: capture(calls),
      },
    });

    await client.responses.create({ model: "m", input: "hi", stream: false });

    expect(calls[0].url).toBe("http://ollama.internal/v1/chat/completions");
    expect(calls[0].headers.authorization).toBe("Bearer k");
  });

  it("reaches ollama by host, appending the compat path", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "ollama",
      config: { host: "http://localhost:11434", fetch: capture(calls) },
    });

    await client.responses.create({
      model: "qwen3",
      input: "hi",
      stream: false,
    });

    expect(calls[0].url).toBe("http://localhost:11434/v1/chat/completions");
  });

  it("leaves a host that already carries the version segment alone", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "ollama",
      config: { host: "http://ollama.internal/v1/", fetch: capture(calls) },
    });

    await client.responses.create({
      model: "qwen3",
      input: "hi",
      stream: false,
    });

    expect(calls[0].url).toBe("http://ollama.internal/v1/chat/completions");
  });

  it("lets baseUrl win over host", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "ollama",
      config: {
        host: "http://ignored:11434",
        baseUrl: "http://explicit/v1",
        fetch: capture(calls),
      },
    });

    await client.responses.create({
      model: "qwen3",
      input: "hi",
      stream: false,
    });

    expect(calls[0].url).toBe("http://explicit/v1/chat/completions");
  });

  it('builds the native NDJSON adapter on api: "native"', async () => {
    const calls: any[] = [];
    const nativeFetch = (async (url: any, init: any) => {
      calls.push({ url: String(url), body: JSON.parse(init.body) });
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: "qwen3",
          created_at: "2024-01-01T00:00:00Z",
          message: { role: "assistant", content: "native ok" },
          done: true,
          prompt_eval_count: 1,
          eval_count: 1,
        }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    const client = new ResponsesClient({
      source: "ollama",
      config: {
        api: "native",
        host: "http://localhost:11434",
        fetch: nativeFetch,
      },
    });

    const resp = await client.responses.create({
      model: "qwen3",
      input: "hi",
      stream: false,
    });

    expect(calls[0].url).toBe("http://localhost:11434/api/chat");
    expect(resp.output_text).toBe("native ok");
  });

  it("passes openRouter-only config through to its adapter", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openRouter",
      config: {
        apiKey: "or-b",
        provider: { order: ["Anthropic"], allow_fallbacks: false },
        appTitle: "app",
        fetch: capture(calls),
      },
    });

    await client.responses.create({ model: "m", input: "hi", stream: false });

    expect(calls[0].url).toBe("https://openrouter.ai/api/v1/chat/completions");
    expect(calls[0].body.provider).toEqual({
      order: ["Anthropic"],
      allow_fallbacks: false,
    });
    expect(calls[0].headers["x-title"]).toBe("app");
  });

  it("exposes the source it was built for", () => {
    const client = new ResponsesClient({
      source: "openRouter",
      config: { apiKey: "or-b" },
    });

    expect(client.source).toBe("openRouter");
  });

  it("routes embeddings to the same backend, stripping the signal", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "ollama",
      config: {
        host: "http://o",
        fetch: capture(calls, {
          object: "list",
          data: [{ object: "embedding", embedding: [0.1], index: 0 }],
          model: "bge-m3",
          usage: { prompt_tokens: 1, total_tokens: 1 },
        }),
      },
    });

    await client.embeddings.create({
      model: "bge-m3",
      input: "hi",
      signal: new AbortController().signal,
    });

    expect(calls[0].url).toBe("http://o/v1/embeddings");
    expect(calls[0].body).not.toHaveProperty("signal");
  });

  it("says so when the backend has no embeddings support", async () => {
    class NoEmbeddings extends ResponsesClient {
      protected override createBackend(): BackendAdapter {
        return {
          name: "bare",
          mode: "completions",
          complete: async () => ({}) as any,
        };
      }
    }
    const client = new NoEmbeddings({ source: "ollama", config: {} });

    await expect(
      client.embeddings.create({ model: "bge-m3", input: "hi" }),
    ).rejects.toThrow(/does not support embeddings/);
  });

  it("builds the openAI store from the provider credentials", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        fetch: capture(calls, { id: "conv_1", object: "conversation" }),
      },
      store: true,
      store_client: "openAI",
    });

    const convo = await client.conversations.create();

    expect(convo.id).toBe("conv_1");
    expect(calls[0].url).toBe("https://api.openai.com/v1/conversations");
    expect(calls[0].headers.Authorization).toBe("Bearer sk-a");
  });

  it("constructs an S3 store without loading the AWS SDK up front", () => {
    const client = new ResponsesClient({
      source: "ollama",
      config: { baseUrl: "http://o/v1" },
      store: true,
      store_client: "S3",
      store_config: { bucket: "b", prefix: "p" },
    });

    expect(client).toBeInstanceOf(ResponsesClient);
  });

  it("persists the turn through the store it built", async () => {
    const hits: string[] = [];
    const items: any[] = [];
    const memStore: Store = {
      createConversation: async () => ({
        id: "conv_mem",
        object: "conversation",
        created_at: 1,
        metadata: null,
      }),
      getConversation: async () => ({
        id: "conv_mem",
        object: "conversation",
        created_at: 1,
        metadata: null,
      }),
      updateConversation: async () => null,
      deleteConversation: async () => ({ id: "conv_mem", deleted: true }),
      appendItems: async (_c, i) => {
        hits.push("append");
        items.push(...i);
      },
      listItems: async () => ({ items, hasMore: false }),
      getItem: async () => null,
      deleteItem: async () => ({ id: "x", deleted: true }),
      saveResponse: async () => {
        hits.push("saveResponse");
      },
      getResponse: async () => null,
      deleteResponse: async () => ({ id: "x", deleted: true }),
    };

    class MemClient extends ResponsesClient {
      protected override createStore(): Store {
        return memStore;
      }
    }

    const calls: any[] = [];
    const client = new MemClient({
      source: "ollama",
      config: { host: "http://o", fetch: capture(calls) },
      store: true,
      store_client: "S3",
      store_config: { bucket: "unused" },
    });

    await client.responses.create({
      model: "m",
      input: "hi",
      conversation: "conv_mem",
      stream: false,
    });

    expect(hits).toEqual(["append", "saveResponse"]);
  });

  /** A store that works except for the one method named. */
  const brokenStore = (
    fails: "appendItems" | "saveResponse",
    err: Error,
  ): Store => {
    const conv = {
      id: "conv_mem",
      object: "conversation" as const,
      created_at: 1,
      metadata: null,
    };
    return {
      createConversation: async () => conv,
      getConversation: async () => conv,
      updateConversation: async () => conv,
      deleteConversation: async () => ({ id: "conv_mem", deleted: true }),
      appendItems: async () => {
        if (fails === "appendItems") throw err;
      },
      listItems: async () => ({ items: [], hasMore: false }),
      getItem: async () => null,
      deleteItem: async () => ({ id: "x", deleted: true }),
      saveResponse: async () => {
        if (fails === "saveResponse") throw err;
      },
      getResponse: async () => null,
      deleteResponse: async () => ({ id: "x", deleted: true }),
    };
  };

  const clientWithStore = (store: Store, fetchImpl?: typeof fetch) => {
    class BrokenStoreClient extends ResponsesClient {
      protected override createStore(): Store {
        return store;
      }
    }
    return new BrokenStoreClient({
      source: "ollama",
      config: { host: "http://o", fetch: fetchImpl ?? capture([]) },
      store: true,
      store_client: "S3",
      store_config: { bucket: "unused" },
    });
  };

  /** A chat-completions SSE stream that says "ok" and stops. */
  const sseFetch = () => {
    const chunk = (delta: any, finish: string | null) =>
      `data: ${JSON.stringify({
        id: "chatcmpl-1",
        object: "chat.completion.chunk",
        created: 1,
        model: "m",
        choices: [{ index: 0, delta, finish_reason: finish }],
      })}\n\n`;
    const sse =
      chunk({ role: "assistant", content: "ok" }, null) +
      chunk({}, "stop") +
      "data: [DONE]\n\n";
    return (async () =>
      new Response(sse, { status: 200 })) as unknown as typeof fetch;
  };

  it("keeps a completed turn when the store refuses the write", async () => {
    const boom = new Error("S3 503");
    const client = clientWithStore(brokenStore("appendItems", boom));

    const err = await client.responses
      .create({
        model: "m",
        input: "hi",
        conversation: "conv_mem",
        stream: false,
      })
      .then(
        () => null,
        (e) => e,
      );

    // The model turn succeeded and was billed. Reporting it as failed, or
    // dropping its output, would make the tokens unrecoverable.
    expect(err).toBeInstanceOf(StorePersistenceError);
    expect(err.response.status).toBe("completed");
    expect(err.response.output_text).toBe("ok");
    expect(err.cause).toBe(boom);
    expect(err.message).toContain("S3 503");
  });

  it("still reports a model failure as a failed response", async () => {
    const client = new ResponsesClient({
      source: "ollama",
      config: {
        host: "http://o",
        fetch: (async () => ({
          ok: false,
          status: 500,
          text: async () => "upstream exploded",
          json: async () => ({}),
        })) as unknown as typeof fetch,
      },
    });

    const err = await client.responses
      .create({ model: "m", input: "hi", stream: false })
      .then(
        () => null,
        (e) => e,
      );

    expect(err).not.toBeInstanceOf(StorePersistenceError);
    expect(err.response.status).toBe("failed");
  });

  it("delivers the stream in full before a store failure is reported", async () => {
    const boom = new Error("S3 503");
    const client = clientWithStore(
      brokenStore("saveResponse", boom),
      sseFetch(),
    );

    const stream = await client.responses.create({
      model: "m",
      input: "hi",
      conversation: "conv_mem",
      stream: true,
    });
    const events: any[] = [];
    for await (const ev of stream) events.push(ev);

    // The turn reached the consumer, so the event log says so. Only the
    // promise — which covers persistence too — reports the failure.
    const completed = events.at(-1);
    expect(completed.type).toBe("response.completed");
    expect(completed.response.status).toBe("completed");
    expect(completed.response.output_text).toBe("ok");
    expect(events.some((e) => e.type === "response.failed")).toBe(false);

    const err = await stream.finalResponse().then(
      () => null,
      (e) => e,
    );
    expect(err).toBeInstanceOf(StorePersistenceError);
    expect(err.response.output_text).toBe("ok");
    expect(err.cause).toBe(boom);
  });

  it("builds a local file store from store_config.dir", async () => {
    const dir = `${tmpdir()}/r2c-test-${process.pid}`;
    const client = new ResponsesClient({
      source: "ollama",
      config: { host: "http://o" },
      store: true,
      store_client: "local",
      store_config: { dir },
    });

    const convo = await client.conversations.create({ metadata: { a: "b" } });
    expect(await client.conversations.get(convo.id)).toMatchObject({
      id: convo.id,
      metadata: { a: "b" },
    });

    await rm(dir, { recursive: true, force: true });
  });

  it("runs without persistence when store is absent", async () => {
    const client = new ResponsesClient({
      source: "ollama",
      config: { baseUrl: "http://o/v1" },
    });

    await expect(client.conversations.create()).rejects.toThrow(
      /requires a store/,
    );
  });

  it("serves conversation CRUD store-only, with no source or credentials", async () => {
    const dir = `${tmpdir()}/r2c-storeonly-${process.pid}`;
    const client = new ResponsesClient({
      store: true,
      store_client: "local",
      store_config: { dir },
    });

    expect(client.source).toBeUndefined();

    const convo = await client.conversations.create({ metadata: { a: "b" } });
    await client.conversations.items.append(convo.id, [
      { type: "message", role: "user", content: "hi" },
    ]);
    const { data } = await client.conversations.items.list(convo.id);
    expect(data).toHaveLength(1);
    expect(await client.conversations.get(convo.id)).toMatchObject({
      metadata: { a: "b" },
    });
    expect(await client.responses.get("resp_missing")).toBeNull();

    await rm(dir, { recursive: true, force: true });
  });

  it("says which half is missing when a store-only client is asked to call a model", async () => {
    const dir = `${tmpdir()}/r2c-storeonly-calls-${process.pid}`;
    const client = new ResponsesClient({
      store: true,
      store_client: "local",
      store_config: { dir },
    });

    await expect(
      client.responses.create({ model: "m", input: "hi", stream: false }),
    ).rejects.toThrow(/`responses.create` requires a source/);
    await expect(
      client.embeddings.create({ model: "m", input: "hi" }),
    ).rejects.toThrow(/`embeddings.create` requires a source/);

    await rm(dir, { recursive: true, force: true });
  });

  it("refuses a client with neither a source nor a store", () => {
    expect(
      // Typed away at compile time; guarded here for JS callers.
      () => new ResponsesClient({} as any),
    ).toThrow(/needs a `source`.*a store to read from, or both/);
  });

  it("rejects the openAI store client on another source", () => {
    expect(
      () =>
        new ResponsesClient({
          source: "ollama",
          config: { baseUrl: "http://o/v1" },
          store: true,
          // Typed away at compile time; guarded here for JS callers.
          store_client: "openAI",
        } as any),
    ).toThrow(/only available for source "openAI"/);
  });

  it("requires store_client when store is on, and a bucket for S3", () => {
    expect(
      () =>
        new ResponsesClient({
          source: "ollama",
          config: { baseUrl: "http://o/v1" },
          store: true,
        } as any),
    ).toThrow(/requires `store_client`/);

    expect(
      () =>
        new ResponsesClient({
          source: "ollama",
          config: { baseUrl: "http://o/v1" },
          store: true,
          store_client: "S3",
        } as any),
    ).toThrow(/needs `store_config.bucket`/);
  });

  it("requires an api key for the hosted sources", () => {
    expect(() => new ResponsesClient({ source: "openAI", config: {} })).toThrow(
      /source "openAI" requires `config.apiKey`/,
    );
    expect(
      () => new ResponsesClient({ source: "openRouter", config: {} }),
    ).toThrow(/source "openRouter" requires `config.apiKey`/);
    // A custom baseUrl is the documented route to keyless OpenAI-compatible
    // servers, so the key requirement lifts there — but not for openRouter.
    expect(
      () =>
        new ResponsesClient({
          source: "openAI",
          config: { baseUrl: "http://localhost:8000/v1" },
        }),
    ).not.toThrow();
    expect(
      () =>
        new ResponsesClient({
          source: "openRouter",
          config: { baseUrl: "http://gateway.internal/v1" } as any,
        }),
    ).toThrow(/source "openRouter" requires `config.apiKey`/);
  });

  it("refuses the openAI store next to a custom baseUrl", () => {
    // The type rejects this pairing (see the compile-time block below); the
    // runtime guard catches JS callers.
    expect(
      () =>
        new ResponsesClient({
          source: "openAI",
          config: {
            apiKey: "sk-a",
            // A generic OpenAI-compatible server — no Conversations API.
            baseUrl: "http://vllm.internal:8000/v1",
          },
          store: true,
          store_client: "openAI",
        } as any),
    ).toThrow(/not available with a custom `config.baseUrl`/);
  });

  it("reaches a keyless server without an authorization header", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        baseUrl: "http://localhost:8000/v1",
        maxTokensParam: "max_tokens",
        fetch: capture(calls),
      },
    });

    await client.responses.create({ model: "m", input: "hi", stream: false });

    expect(calls[0].url).toBe("http://localhost:8000/v1/chat/completions");
    expect(calls[0].headers).not.toHaveProperty("authorization");
  });

  it("lets a subclass override how the backend is built", async () => {
    const hits: string[] = [];
    class Custom extends ResponsesClient {
      protected override createBackend(): any {
        return {
          name: "custom",
          mode: "completions",
          complete: async (req: any) => {
            hits.push("custom");
            return {
              id: "c",
              object: "chat.completion",
              created: 1,
              model: req.model,
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: "x" },
                  finish_reason: "stop",
                },
              ],
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
              },
            };
          },
        };
      }
    }

    const client = new Custom({ source: "openAI", config: { apiKey: "sk-a" } });
    await client.responses.create({ model: "m", input: "hi", stream: false });

    expect(hits).toEqual(["custom"]);
  });
});

describe("option typing (compile-time)", () => {
  // `tsc -p tsconfig.test.json` (npm run typecheck) is what enforces these:
  // every `@ts-expect-error` below fails the build if the type stops rejecting
  // the combination. At runtime the block only proves the accepted shapes are
  // constructible.
  const accepts = (o: ResponsesClientOptions) => o;

  it("accepts the valid combinations", () => {
    accepts({ source: "openAI", config: { apiKey: "k" } });
    accepts({ source: "ollama", config: { baseUrl: "http://o/v1" } });
    accepts({ source: "ollama", config: { host: "http://localhost:11434" } });
    accepts({
      source: "ollama",
      config: {},
      store: true,
      store_client: "local",
      store_config: { dir: "./.data" },
    });
    accepts({ source: "ollama", config: { api: "native", host: "http://o" } });
    // Store-only: no source, no config, and a store to read from.
    accepts({ store: true, store_client: "S3", store_config: { bucket: "b" } });
    accepts({
      store: true,
      store_client: "local",
      store_config: { dir: "./.data" },
    });
    accepts({
      source: "openRouter",
      config: { apiKey: "k", provider: { sort: "price" } },
      store: true,
      store_client: "S3",
      store_config: { bucket: "b" },
    });
    accepts({
      source: "openAI",
      config: { apiKey: "k" },
      store: true,
      store_client: "openAI",
    });
    // A compat server (custom baseUrl) can persist to S3/local, just not to
    // OpenAI's store.
    accepts({
      source: "openAI",
      config: { apiKey: "k", baseUrl: "http://vllm:8000/v1" },
      store: true,
      store_client: "S3",
      store_config: { bucket: "b" },
    });
    expect(true).toBe(true);
  });

  it("rejects mismatched config and store combinations", () => {
    accepts({
      source: "openAI",
      // @ts-expect-error — `provider` is an openRouter-only key.
      config: { apiKey: "k", provider: { sort: "price" } },
    });
    // @ts-expect-error — store_client is not accepted without `store: true`.
    accepts({ source: "openAI", config: { apiKey: "k" }, store_client: "S3" });
    // @ts-expect-error — the "openAI" store client requires source "openAI".
    accepts({
      source: "ollama",
      config: {},
      store: true,
      store_client: "openAI",
    });
    // @ts-expect-error — a custom baseUrl names a compat server with no Conversations API, so the "openAI" store is rejected.
    accepts({
      source: "openAI",
      config: { apiKey: "k", baseUrl: "http://vllm:8000/v1" },
      store: true,
      store_client: "openAI",
    });
    // @ts-expect-error — S3 needs `store_config.bucket`.
    accepts({ source: "ollama", config: {}, store: true, store_client: "S3" });
    // @ts-expect-error — the local store needs `store_config.dir`.
    accepts({
      source: "ollama",
      config: {},
      store: true,
      store_client: "local",
    });
    // @ts-expect-error — openAI's config is required.
    accepts({ source: "openAI" });
    // @ts-expect-error — a named source always requires its config, ollama included.
    accepts({ source: "ollama" });
    // @ts-expect-error — `config` without a `source` has nothing to configure.
    accepts({
      config: { apiKey: "k" },
      store: true,
      store_client: "S3",
      store_config: { bucket: "b" },
    });
    // @ts-expect-error — neither a source nor a store: every method would throw.
    accepts({});
    // @ts-expect-error — a store-only client needs a real store, not `store: false`.
    accepts({ store: false });
    // @ts-expect-error — the openAI store client has no source to borrow credentials from.
    accepts({ store: true, store_client: "openAI" });
    // @ts-expect-error — `api` is an ollama-only key.
    accepts({ source: "openAI", config: { apiKey: "k", api: "native" } });
    // @ts-expect-error — `host` is an ollama-only key.
    accepts({ source: "openAI", config: { apiKey: "k", host: "http://x" } });
    // @ts-expect-error — snake_case keys are not accepted; the config is camelCase.
    accepts({ source: "openAI", config: { api_key: "k" } });
    // @ts-expect-error — snake_case keys are not accepted; the config is camelCase.
    accepts({ source: "ollama", config: { base_url: "http://o/v1" } });
    accepts({
      source: "openRouter",
      // @ts-expect-error — OpenRouterAdapter has no maxTokensParam option.
      config: { apiKey: "k", maxTokensParam: "max_tokens" },
    });
    // @ts-expect-error — OpenAICompatAdapter has no provider-routing option.
    accepts({
      source: "openAI",
      config: { apiKey: "k", usageAccounting: true },
    });
    accepts({
      source: "ollama",
      // @ts-expect-error — native ollama has no baseUrl; it is addressed by host.
      config: { api: "native", baseUrl: "http://o/v1" },
    });
    // @ts-expect-error — unknown source.
    accepts({ source: "anthropic", config: {} });
    expect(true).toBe(true);
  });
});

describe("history-field typing (compile-time)", () => {
  // Same deal as the block above: `npm run typecheck` is the assertion. The
  // client's store flag is inferred from the `store` literal, and a client
  // with no store rejects the two fields only a store can resolve — which
  // `resolveHistory` would otherwise have to throw on at the call.
  const stored = new ResponsesClient({
    source: "openAI",
    config: { apiKey: "k" },
    store: true,
    store_client: "openAI",
  });
  const storeless = new ResponsesClient({
    source: "openAI",
    config: { apiKey: "k" },
    store: false,
  });

  it("takes the history fields on a stored client", () => {
    const calls = [
      () => stored.responses.create({ model: "m", input: "i", conversation: "conv_1" }),
      () => stored.responses.create({ model: "m", input: "i", conversation: { id: "conv_1" } }),
      () => stored.responses.create({ model: "m", input: "i", previous_response_id: "resp_1" }),
      // The request-level flag is a different switch and stays available:
      // read the conversation, don't persist this turn.
      () =>
        stored.responses.create({
          model: "m",
          input: "i",
          conversation: "conv_1",
          store: false,
        }),
    ];
    expect(calls).toHaveLength(4);
  });

  it("rejects the history fields on a storeless client", () => {
    const calls = [
      // @ts-expect-error — `conversation` needs a store on the client.
      () => storeless.responses.create({ model: "m", input: "i", conversation: "conv_1" }),
      // @ts-expect-error — so does `previous_response_id`.
      () => storeless.responses.create({ model: "m", input: "i", previous_response_id: "resp_1" }),
      // @ts-expect-error — streaming is typed off the same request.
      () => storeless.responses.create({ model: "m", input: "i", conversation: "c", stream: true }),
    ];
    // Everything else about the storeless client is unchanged.
    const plain = () =>
      storeless.responses.create({ model: "m", input: "i", store: false });
    expect(calls).toHaveLength(3);
    expect(typeof plain).toBe("function");
  });

  it("stays permissive where the store flag isn't known", () => {
    // A bare annotation (or a subclass) keeps `boolean`, so both shapes type
    // and the store check remains the runtime one.
    const anyClient: ResponsesClient = stored;
    const alsoAny: ResponsesClient = storeless;
    const call = () =>
      anyClient.responses.create({ model: "m", input: "i", conversation: "c" });
    expect([alsoAny, call]).toHaveLength(2);
  });
});

describe("native /responses turns keep the provider's id", () => {
  /**
   * Answers `POST /responses` (and the store's `GET /responses/<id>` and
   * `POST /conversations`) the way OpenAI does: with ids it issued itself.
   */
  const openAIFetch = (calls: any[], upstreamId = "resp_upstream_1") => {
    const stored: Record<string, unknown> = {};
    return (async (url: any, init: any) => {
      const u = String(url);
      const method = init?.method ?? "GET";
      calls.push({
        url: u,
        method,
        body: init?.body ? JSON.parse(init.body) : undefined,
      });

      if (method === "POST" && u.endsWith("/responses")) {
        const resp = {
          id: upstreamId,
          object: "response",
          created_at: 1,
          status: "completed",
          model: "m",
          output: [
            {
              type: "message",
              id: "msg_1",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "ok", annotations: [] }],
            },
          ],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        };
        stored[upstreamId] = resp;
        return new Response(JSON.stringify(resp), { status: 200 });
      }
      if (method === "GET" && u.includes("/responses/")) {
        const id = decodeURIComponent(u.split("/responses/")[1]);
        return stored[id]
          ? new Response(JSON.stringify(stored[id]), { status: 200 })
          : new Response("not found", { status: 404 });
      }
      return new Response("{}", { status: 200 });
    }) as unknown as typeof fetch;
  };

  it("returns the upstream id instead of a locally generated one", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: openAIFetch(calls),
      },
      store: true,
      store_client: "openAI",
    });

    const resp = await client.responses.create({
      model: "m",
      input: "hi",
      stream: false,
    });

    expect(resp.id).toBe("resp_upstream_1");
    expect(calls[0].url).toBe("https://api.openai.com/v1/responses");
  });

  it("keeps the upstream terminal status instead of stamping completed", async () => {
    const fetchMock = (async () =>
      new Response(
        JSON.stringify({
          id: "resp_up",
          object: "response",
          created_at: 1,
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
          model: "m",
          output: [],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        }),
        { status: 200 },
      )) as unknown as typeof fetch;
    const client = new ResponsesClient({
      source: "openAI",
      config: { apiKey: "sk-a", endpoint: "responses", fetch: fetchMock },
    });

    const resp = await client.responses.create({
      model: "m",
      input: "hi",
      stream: false,
    });

    expect(resp.status).toBe("incomplete");
    expect(resp.incomplete_details).toEqual({ reason: "max_output_tokens" });
  });

  it("throws on an upstream failed response even without an error object", async () => {
    const fetchMock = (async () =>
      new Response(
        JSON.stringify({
          id: "resp_up",
          object: "response",
          created_at: 1,
          status: "failed",
          error: null,
          model: "m",
          output: [],
        }),
        { status: 200 },
      )) as unknown as typeof fetch;
    const client = new ResponsesClient({
      source: "openAI",
      config: { apiKey: "sk-a", endpoint: "responses", fetch: fetchMock },
    });

    await expect(
      client.responses.create({ model: "m", input: "hi", stream: false }),
    ).rejects.toThrow(/Upstream \/responses failed/);
  });

  it("passes input_file parts through to /responses untouched", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: openAIFetch(calls),
      },
    });

    await client.responses.create({
      model: "m",
      input: [
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: "Summarize this." },
            {
              type: "input_file",
              file_url: "https://cdn.example/contract.pdf",
            },
          ],
        },
      ],
      stream: false,
    });

    // No translation on this path: file_url is a native /responses field.
    expect(calls[0].body.input[0].content[1]).toEqual({
      type: "input_file",
      file_url: "https://cdn.example/contract.pdf",
    });
  });

  it("forwards an explicit store opt-out to the upstream /responses body", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: openAIFetch(calls),
      },
    });

    await client.responses.create({
      model: "m",
      input: "hi",
      store: false,
      stream: false,
    });
    // Without the caller's opt-out, the provider's own default applies.
    await client.responses.create({ model: "m", input: "hi", stream: false });

    expect(calls[0].body.store).toBe(false);
    expect(calls[1].body).not.toHaveProperty("store");
  });

  it("resolves previous_response_id through the openAI store", async () => {
    const calls: any[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: openAIFetch(calls),
      },
      store: true,
      store_client: "openAI",
    });

    const first = await client.responses.create({
      model: "m",
      input: "teal",
      stream: false,
    });
    const second = await client.responses.create({
      model: "m",
      previous_response_id: first.id,
      input: "which colour?",
      stream: false,
    });

    // The prior turn was fetched back from OpenAI and folded into the input.
    expect(
      calls.some(
        (c) => c.method === "GET" && c.url.endsWith(`/responses/${first.id}`),
      ),
    ).toBe(true);
    const lastPost = calls.filter((c) => c.method === "POST").at(-1);
    expect(JSON.stringify(lastPost.body.input)).toContain("ok");
    expect(second.id).toBe("resp_upstream_1");
  });

  it("announces the upstream id on the stream's first lifecycle event", async () => {
    const sse = [
      `event: response.created\ndata: ${JSON.stringify({
        type: "response.created",
        sequence_number: 0,
        response: {
          id: "resp_upstream_stream",
          object: "response",
          status: "in_progress",
        },
      })}\n\n`,
      `event: response.output_text.delta\ndata: ${JSON.stringify({
        type: "response.output_text.delta",
        sequence_number: 1,
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        delta: "ok",
      })}\n\n`,
      `event: response.completed\ndata: ${JSON.stringify({
        type: "response.completed",
        sequence_number: 2,
        response: {
          id: "resp_upstream_stream",
          object: "response",
          status: "completed",
          output: [],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        },
      })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");

    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: (async () =>
          new Response(sse, { status: 200 })) as unknown as typeof fetch,
      },
    });

    const stream = await client.responses.create({
      model: "m",
      input: "hi",
      stream: true,
    });
    const events: any[] = [];
    for await (const ev of stream) events.push(ev);
    const final = await stream.finalResponse();

    expect(events[0].type).toBe("response.created");
    expect(events[0].response.id).toBe("resp_upstream_stream");
    expect(events[1].type).toBe("response.in_progress");
    expect(final.id).toBe("resp_upstream_stream");
    // One id for the whole stream, lifecycle events included.
    const ids = new Set(
      events.filter((e) => e.response).map((e) => e.response.id),
    );
    expect([...ids]).toEqual(["resp_upstream_stream"]);
  });

  it("ends the stream with response.incomplete when the upstream turn was truncated", async () => {
    const sse = [
      `event: response.created\ndata: ${JSON.stringify({
        type: "response.created",
        sequence_number: 0,
        response: {
          id: "resp_up_trunc",
          object: "response",
          status: "in_progress",
        },
      })}\n\n`,
      `event: response.output_text.delta\ndata: ${JSON.stringify({
        type: "response.output_text.delta",
        sequence_number: 1,
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        delta: "ok",
      })}\n\n`,
      `event: response.incomplete\ndata: ${JSON.stringify({
        type: "response.incomplete",
        sequence_number: 2,
        response: {
          id: "resp_up_trunc",
          object: "response",
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
          output: [],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        },
      })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");

    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        endpoint: "responses",
        fetch: (async () =>
          new Response(sse, { status: 200 })) as unknown as typeof fetch,
      },
    });

    const stream = await client.responses.create({
      model: "m",
      input: "hi",
      stream: true,
    });
    const events: any[] = [];
    for await (const ev of stream) events.push(ev);
    const final = await stream.finalResponse();

    // Exactly one terminal event, and it is response.incomplete.
    const terminal = events.filter((e) =>
      ["response.completed", "response.incomplete", "response.failed"].includes(
        e.type,
      ),
    );
    expect(terminal.map((e) => e.type)).toEqual(["response.incomplete"]);
    expect(terminal[0].response.status).toBe("incomplete");
    expect(final.status).toBe("incomplete");
    expect(final.incomplete_details).toEqual({ reason: "max_output_tokens" });
    // Usage arrives on the incomplete event and must still be harvested.
    expect(final.usage).toEqual({
      input_tokens: 1,
      output_tokens: 1,
      total_tokens: 2,
    });
  });

  it("explains an unresolvable previous_response_id on a read-through store", async () => {
    const client = new ResponsesClient({
      source: "openAI",
      // Default endpoint: the turn is synthesized here, so OpenAI never saw it.
      config: {
        apiKey: "sk-a",
        fetch: (async () =>
          new Response("not found", {
            status: 404,
          })) as unknown as typeof fetch,
      },
      store: true,
      store_client: "openAI",
    });

    await expect(
      client.responses.create({
        model: "m",
        previous_response_id: "resp_local",
        input: "hi",
        stream: false,
      }),
    ).rejects.toThrow(/only ids the provider issued resolve/);
  });

  it("explains an unknown conversation id instead of attempting an auto-create", async () => {
    const attempted: string[] = [];
    const client = new ResponsesClient({
      source: "openAI",
      config: {
        apiKey: "sk-a",
        fetch: (async (url: any, init: any) => {
          attempted.push(`${init?.method ?? "GET"} ${String(url)}`);
          return new Response("not found", { status: 404 });
        }) as unknown as typeof fetch,
      },
      store: true,
      store_client: "openAI",
    });

    await expect(
      client.responses.create({
        model: "m",
        input: "hi",
        conversation: "conv_mine",
        stream: false,
      }),
    ).rejects.toThrow(/does not assign conversation ids/);

    // The lookup happened; no doomed POST /conversations followed it.
    expect(attempted).toEqual([
      "GET https://api.openai.com/v1/conversations/conv_mine",
    ]);
  });
});

describe("request-level maxIterations", () => {
  const neverFetch = (async () => {
    throw new Error("backend must not be contacted");
  }) as unknown as typeof fetch;

  it.each([false, true])(
    "rejects an invalid value before contacting the backend (stream: %s)",
    async (stream) => {
      const client = new ResponsesClient({
        source: "openAI",
        config: { apiKey: "sk-a", fetch: neverFetch },
      });

      await expect(
        client.responses.create({
          model: "m",
          input: "hi",
          maxIterations: 0,
          stream,
        }),
      ).rejects.toThrow("`maxIterations` must be a positive integer");
    },
  );
});
