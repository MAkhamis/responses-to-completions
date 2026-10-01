import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createStoreForClient } from "./from-source.js";
import {
  OpenAIConversationStore,
  OpenAIStoreError,
  OpenAIStoreTimeoutError,
} from "./openai.js";
import type { ConversationItem } from "./store.js";

type Call = { url: string; method: string; body: any };

const mockFetch = (
  calls: Call[],
  routes: (
    url: string,
    method: string,
    body: any,
  ) => { status?: number; json: any },
) =>
  (async (url: any, init: any) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(init.body) : undefined;
    calls.push({ url: String(url), method, body });
    const { status = 200, json } = routes(String(url), method, body);
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => json,
      text: async () => JSON.stringify(json),
    };
  }) as unknown as typeof fetch;

const msg = (text: string, id?: string): ConversationItem =>
  ({
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text }],
    ...(id ? { id } : {}),
  }) as ConversationItem;

describe("OpenAIConversationStore", () => {
  it("creates a conversation upstream and returns OpenAI's id", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({
        json: { id: "conv_openai_1", object: "conversation", created_at: 1 },
      })),
    });

    const convo = await store.createConversation({});

    expect(convo.id).toBe("conv_openai_1");
    expect(calls[0].method).toBe("POST");
    expect(calls[0].url).toBe("https://api.openai.com/v1/conversations");
  });

  it("refuses a caller-supplied id instead of silently returning a different one", async () => {
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch([], () => ({ json: {} })),
    });

    await expect(store.createConversation({ id: "conv_mine" })).rejects.toThrow(
      /OpenAI assigns conversation ids/,
    );
  });

  it("reads the whole conversation in chronological order, paging past the default page size", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      pageSize: 2,
      fetch: mockFetch(calls, (url) => {
        if (url.includes("after=i2")) {
          return {
            json: { data: [msg("c", "i3")], has_more: false, last_id: "i3" },
          };
        }
        return {
          json: {
            data: [msg("a", "i1"), msg("b", "i2")],
            has_more: true,
            last_id: "i2",
          },
        };
      }),
    });

    const { items, hasMore } = await store.listItems("conv_1");

    expect(items).toHaveLength(3);
    expect(hasMore).toBe(false);
    expect(calls[0].url).toContain("order=asc");
    expect(calls[1].url).toContain("after=i2");
  });

  it("asks OpenAI to include input_image URLs on every listed page and on a single item", async () => {
    // OpenAI omits `image_url` from listed `input_image` parts unless asked,
    // and a part with neither `image_url` nor `file_id` is rejected the moment
    // the history is replayed as `input`.
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      pageSize: 1,
      fetch: mockFetch(calls, (url) => {
        if (url.includes("/items/")) return { json: msg("a", "i1") };
        if (url.includes("after=i1")) {
          return {
            json: { data: [msg("b", "i2")], has_more: false, last_id: "i2" },
          };
        }
        return {
          json: { data: [msg("a", "i1")], has_more: true, last_id: "i1" },
        };
      }),
    });

    await store.listItems("conv_1");
    await store.listItems("conv_1", { limit: 5 });
    await store.getItem("conv_1", "i1");

    expect(calls).toHaveLength(4);
    for (const call of calls) {
      expect(decodeURIComponent(call.url)).toContain(
        "include[]=message.input_image.image_url",
      );
    }
    // The pagination cursor and page size still ride alongside it.
    expect(decodeURIComponent(calls[1].url)).toContain("after=i1");
    expect(decodeURIComponent(calls[2].url)).toContain("limit=5");
  });

  it("returns a single page verbatim when the caller sets a limit", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({
        json: { data: [msg("a", "i1")], has_more: true, last_id: "i1" },
      })),
    });

    const { items, hasMore } = await store.listItems("conv_1", { limit: 1 });

    expect(items).toHaveLength(1);
    expect(hasMore).toBe(true);
    expect(calls).toHaveLength(1);
  });

  it("strips locally-generated item ids before appending", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: { data: [] } })),
    });

    await store.appendItems("conv_1", [msg("hello", "msg_local_1")]);

    expect(calls[0].method).toBe("POST");
    expect(calls[0].body.items[0]).not.toHaveProperty("id");
    expect(calls[0].body.items[0].content[0].text).toBe("hello");
  });

  it("keeps the id on reasoning items, which require one", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: { data: [] } })),
    });

    await store.appendItems("conv_1", [
      {
        type: "reasoning",
        id: "rs_abc",
        summary: [{ type: "summary_text", text: "thought" }],
      } as ConversationItem,
      msg("hello", "msg_local_1"),
    ]);

    expect(calls[0].body.items[0].id).toBe("rs_abc");
    expect(calls[0].body.items[1]).not.toHaveProperty("id");
  });

  it("drops the SDK-private fields OpenAI would reject as unknown", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: { data: [] } })),
    });

    await store.appendItems("conv_1", [
      {
        type: "reasoning",
        id: "rs_abc",
        encrypted_content: "blob",
        model: "gpt-5",
      } as unknown as ConversationItem,
      {
        type: "mcp_list_tools",
        id: "mcpl_1",
        server_label: "docs",
        tools: [],
        error: "connect ECONNREFUSED",
      } as unknown as ConversationItem,
    ]);

    expect(calls[0].body.items[0].encrypted_content).toBe("blob");
    expect(calls[0].body.items[0]).not.toHaveProperty("model");
    expect(calls[0].body.items[1]).not.toHaveProperty("error");
  });

  it("clamps limit and pageSize to OpenAI's 1-100 bound", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      pageSize: 500,
      fetch: mockFetch(calls, () => ({
        json: { data: [msg("a", "i1")], has_more: false, last_id: "i1" },
      })),
    });

    // Explicit oversized limit: one page, clamped, hasMore reports the rest.
    await store.listItems("conv_1", { limit: 200 });
    expect(calls[0].url).toContain("limit=100");

    // Zero/negative limits would 400 too — clamped up to 1.
    await store.listItems("conv_1", { limit: 0 });
    expect(calls[1].url).toContain("limit=1");

    // Full-history paging uses the (clamped) pageSize.
    await store.listItems("conv_1");
    expect(calls[2].url).toContain("limit=100");
  });

  it("splits large appends into 20-item batches, in order", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: { data: [] } })),
    });

    const items = Array.from({ length: 45 }, (_, i) => msg(`m${i}`));
    await store.appendItems("conv_1", items);

    expect(calls).toHaveLength(3);
    expect(calls.map((c) => c.body.items.length)).toEqual([20, 20, 5]);
    expect(calls[0].body.items[0].content[0].text).toBe("m0");
    expect(calls[1].body.items[0].content[0].text).toBe("m20");
    expect(calls[2].body.items[4].content[0].text).toBe("m44");
  });

  it("creates a conversation with oversized initial items by batching the rest", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, (url) =>
        url.endsWith("/conversations")
          ? {
              json: {
                id: "conv_openai_1",
                object: "conversation",
                created_at: 1,
              },
            }
          : { json: { data: [] } },
      ),
    });

    const items = Array.from({ length: 25 }, (_, i) => msg(`m${i}`));
    const convo = await store.createConversation({ items });

    expect(convo.id).toBe("conv_openai_1");
    expect(calls).toHaveLength(2);
    expect(calls[0].url).toBe("https://api.openai.com/v1/conversations");
    expect(calls[0].body.items).toHaveLength(20);
    expect(calls[1].url).toBe(
      "https://api.openai.com/v1/conversations/conv_openai_1/items",
    );
    expect(calls[1].body.items).toHaveLength(5);
    expect(calls[1].body.items[0].content[0].text).toBe("m20");
  });

  it("gives every fetch a signal, including each page", async () => {
    // Reads get a per-attempt signal — it also carries the timeout — that
    // follows the caller's; an append, which never times out, gets the
    // caller's own.
    const signals: Array<AbortSignal | undefined> = [];
    const pages = [
      { data: [msg("a", "i1")], has_more: true, last_id: "i1" },
      { data: [msg("b", "i2")], has_more: false, last_id: "i2" },
    ];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: (async (_url: any, init: any) => {
        signals.push(init?.signal);
        const json =
          init?.method === "POST"
            ? { data: [] }
            : (pages.shift() ?? { data: [] });
        return {
          ok: true,
          status: 200,
          json: async () => json,
          text: async () => JSON.stringify(json),
        };
      }) as unknown as typeof fetch,
    });
    const ac = new AbortController();

    await store.listItems("conv_1", undefined, ac.signal);
    await store.appendItems("conv_1", [msg("hello")], ac.signal);
    await store.getConversation("conv_1", ac.signal);

    expect(signals).toHaveLength(4); // two pages + append + get
    for (const s of signals) expect(s).toBeInstanceOf(AbortSignal);
    expect(signals[2]).toBe(ac.signal); // the append
  });

  it("cancels an in-flight page when the caller aborts, without retrying it", async () => {
    const calls: string[] = [];
    const ac = new AbortController();
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: (async (url: any, init: any) => {
        calls.push(String(url));
        if (calls.length === 1) {
          const json = {
            data: [msg("a", "i1")],
            has_more: true,
            last_id: "i1",
          };
          return {
            ok: true,
            status: 200,
            json: async () => json,
            text: async () => JSON.stringify(json),
          };
        }
        // The second page is in flight when the caller gives up.
        return new Promise((_, reject) => {
          init.signal.addEventListener(
            "abort",
            () => reject(init.signal.reason),
            { once: true },
          );
          ac.abort(new Error("caller gave up"));
        });
      }) as unknown as typeof fetch,
    });

    await expect(
      store.listItems("conv_1", undefined, ac.signal),
    ).rejects.toThrow("caller gave up");
    expect(calls).toHaveLength(2);
  });

  it("keeps the error OpenAI itself defines on an mcp_call item", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: { data: [] } })),
    });

    await store.appendItems("conv_1", [
      {
        type: "mcp_call",
        id: "mcp_1",
        server_label: "docs",
        name: "search",
        arguments: "{}",
        output: null,
        error: "tool failed",
        approval_request_id: null,
      } as unknown as ConversationItem,
    ]);

    expect(calls[0].body.items[0].error).toBe("tool failed");
  });

  it("leaves metadata alone when the patch doesn't mention it", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({
        json: {
          id: "conv_1",
          object: "conversation",
          created_at: 1,
          metadata: { owner: "alice" },
        },
      })),
    });

    const convo = await store.updateConversation("conv_1", {});

    // A write would have sent `metadata: null` and cleared it upstream.
    expect(calls).toHaveLength(1);
    expect(calls[0].method).toBe("GET");
    expect(convo?.metadata).toEqual({ owner: "alice" });
  });

  it("clears metadata when the patch says null", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({
        json: {
          id: "conv_1",
          object: "conversation",
          created_at: 1,
          metadata: null,
        },
      })),
    });

    await store.updateConversation("conv_1", { metadata: null });

    expect(calls[0].method).toBe("POST");
    expect(calls[0].body).toEqual({ metadata: null });
  });

  it("writes the metadata it was given", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({
        json: {
          id: "conv_1",
          object: "conversation",
          created_at: 1,
          metadata: { owner: "bob" },
        },
      })),
    });

    await store.updateConversation("conv_1", { metadata: { owner: "bob" } });

    expect(calls[0].method).toBe("POST");
    expect(calls[0].url).toBe("https://api.openai.com/v1/conversations/conv_1");
    expect(calls[0].body).toEqual({ metadata: { owner: "bob" } });
  });

  it("treats a missing conversation as absent rather than an error", async () => {
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch([], () => ({ status: 404, json: { error: "missing" } })),
    });

    expect(await store.getConversation("conv_gone")).toBeNull();
  });

  it("surfaces non-404 failures", async () => {
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch([], () => ({ status: 401, json: { error: "bad key" } })),
    });

    await expect(store.getConversation("conv_1")).rejects.toThrow(/401/);
  });

  it("reads a bodyless success as done, not as a parse failure", async () => {
    // 204, and 200-with-empty-body, are both valid answers to a DELETE.
    const bodyless = (status: number, body: string) =>
      (async () =>
        ({
          ok: true,
          status,
          text: async () => body,
          json: async () => {
            throw new SyntaxError("Unexpected end of JSON input");
          },
        }) as unknown as Response) as unknown as typeof fetch;

    for (const [status, body] of [
      [204, ""],
      [200, ""],
      [200, "\n"],
    ] as const) {
      const store = new OpenAIConversationStore({
        apiKey: "sk-test",
        fetch: bodyless(status, body),
      });

      expect(await store.deleteConversation("conv_1")).toEqual({
        id: "conv_1",
        deleted: true,
      });
      expect(await store.deleteItem("conv_1", "msg_1")).toEqual({
        id: "msg_1",
        deleted: true,
      });
      expect(await store.deleteResponse("resp_1")).toEqual({
        id: "resp_1",
        deleted: true,
      });
    }
  });

  it("reads a response through from the provider, and 404s as null", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, (url) =>
        url.endsWith("/responses/resp_live")
          ? {
              json: {
                id: "resp_live",
                object: "response",
                status: "completed",
              },
            }
          : { status: 404, json: { error: "not found" } },
      ),
    });

    const found = await store.getResponse("resp_live");
    expect(found?.id).toBe("resp_live");
    expect(calls[0].method).toBe("GET");
    expect(calls[0].url).toBe("https://api.openai.com/v1/responses/resp_live");

    expect(await store.getResponse("resp_missing")).toBeNull();
  });

  it("stops paging when the cursor fails to advance instead of looping", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      // A misbehaving server: always has_more with the same last_id.
      fetch: mockFetch(calls, () => ({
        json: { data: [msg("a", "i1")], has_more: true, last_id: "i1" },
      })),
    });

    const { items, hasMore } = await store.listItems("conv_1");

    expect(calls).toHaveLength(2); // first page + the non-advancing repeat
    expect(items).toHaveLength(1); // the repeated page is not duplicated
    expect(hasMore).toBe(false);
  });

  it("no-ops saveResponse — OpenAI persists what its own endpoint produced", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: mockFetch(calls, () => ({ json: {} })),
    });

    await store.saveResponse({ id: "resp_local" } as any);

    expect(calls).toHaveLength(0);
  });
});

// What api.openai.com's Cloudflare edge answers when the origin times out,
// trimmed from a real page (2026-09-30).
const CLOUDFLARE_504 = `<!DOCTYPE html>
<!--[if lt IE 7]> <html class="no-js ie6 oldie" lang="en-US"> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang="en-US"> <!--<![endif]-->
<head>
<title>api.openai.com | 504: Gateway time-out</title>
<meta charset="UTF-8" />
<link rel="stylesheet" id="cf_styles-css" href="/cdn-cgi/styles/main.css" />
</head>
<body>
<div id="cf-wrapper">
  <h1><span class="inline-block">Gateway time-out</span>
  <span class="code-label">Error code 504</span></h1>
  <p class="text-13">
    <span class="cf-footer-item sm:block sm:mb-1">Cloudflare Ray ID: <strong class="font-semibold">a43247a5be46f282</strong></span>
  </p>
</div>
</body>
</html>`;

type Step =
  | {
      status: number;
      json?: unknown;
      text?: string;
      headers?: Record<string, string>;
    }
  | { reject: unknown }
  | { hang: true }
  | { delayMs: number; status: number; json?: unknown };

/** A fetch that plays `steps` in order, one per request. */
const scripted = (calls: Call[], steps: Step[]) =>
  (async (url: any, init: any) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(init.body) : undefined;
    calls.push({ url: String(url), method, body });
    const step = steps.shift();
    if (!step) throw new Error(`unexpected request #${calls.length}`);
    if ("reject" in step) throw step.reject;
    if ("hang" in step) {
      // Never answers; only an abort ends it, like a stuck upstream.
      return new Promise((_, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(init.signal.reason),
          { once: true },
        );
      });
    }
    if ("delayMs" in step) {
      // Answers late — and, like a real fetch, gives up if aborted first.
      await new Promise((resolve, reject) => {
        setTimeout(resolve, step.delayMs);
        init?.signal?.addEventListener(
          "abort",
          () => reject(init.signal.reason),
          { once: true },
        );
      });
    }
    const text =
      "text" in step && step.text !== undefined
        ? step.text
        : JSON.stringify(step.json ?? {});
    return {
      ok: step.status >= 200 && step.status < 300,
      status: step.status,
      headers: new Headers("headers" in step ? step.headers : {}),
      text: async () => text,
      json: async () => JSON.parse(text),
    };
  }) as unknown as typeof fetch;

const CONV = { id: "conv_1", object: "conversation", created_at: 1 };

/** undici's rejection for a socket error: `TypeError` with the code on `cause`. */
const networkError = (code: string) =>
  new TypeError("fetch failed", {
    cause: Object.assign(new Error(code), { code }),
  });

describe("OpenAIConversationStore retries", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  /** Settles a store call while the fake clock runs every backoff. */
  const settle = async <T>(p: Promise<T>): Promise<T> => {
    const outcome = p.then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    );
    await vi.runAllTimersAsync();
    const result = await outcome;
    if ("error" in result) throw result.error;
    return result.value;
  };

  it("retries a read the gateway timed out, then returns the answer", async () => {
    // The failure that motivated this: the history read before a turn got a
    // Cloudflare 504 page from api.openai.com and failed the whole turn.
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        { status: 504, text: CLOUDFLARE_504 },
        { status: 200, json: CONV },
      ]),
    });

    const convo = await settle(store.getConversation("conv_1"));

    expect(convo?.id).toBe("conv_1");
    expect(calls).toHaveLength(2);
  });

  it("retries every page of a listing on its own", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        {
          status: 200,
          json: { data: [msg("a", "i1")], has_more: true, last_id: "i1" },
        },
        {
          status: 502,
          text: "<html><head><title>502 Bad Gateway</title></head></html>",
        },
        {
          status: 200,
          json: { data: [msg("b", "i2")], has_more: false, last_id: "i2" },
        },
      ]),
    });

    const { items } = await settle(store.listItems("conv_1"));

    expect(items).toHaveLength(2);
    expect(calls).toHaveLength(3);
    expect(calls[1].url).toContain("after=i1");
    expect(calls[2].url).toContain("after=i1");
  });

  it("gives up after maxRetries and reports a gateway page as one readable line", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(
        calls,
        Array.from({ length: 3 }, () => ({
          status: 504,
          text: CLOUDFLARE_504,
        })),
      ),
    });

    const err = await settle(store.listItems("conv_1")).catch((e) => e);

    expect(calls).toHaveLength(3); // the first attempt + 2 retries
    expect(err).toBeInstanceOf(OpenAIStoreError);
    expect(err.status).toBe(504);
    expect(err.attempts).toBe(3);
    expect(err.body).toBe(CLOUDFLARE_504); // the raw page stays available
    expect(err.message).toBe(
      "OpenAI conversations API error 504: api.openai.com | 504: Gateway time-out (HTML error page, Cloudflare Ray ID a43247a5be46f282) [GET /conversations/conv_1/items, 3 attempts]",
    );
  });

  it("does not retry an error that will not change", async () => {
    for (const status of [400, 401, 403, 422]) {
      const calls: Call[] = [];
      const store = new OpenAIConversationStore({
        apiKey: "sk-test",
        fetch: scripted(calls, [
          { status, json: { error: { message: "no" } } },
        ]),
      });

      const err = await settle(store.getConversation("conv_1")).catch((e) => e);

      expect(err).toBeInstanceOf(OpenAIStoreError);
      expect(err.status).toBe(status);
      expect(calls).toHaveLength(1);
    }
  });

  it("follows the server's x-should-retry hint over the status", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        { status: 503, json: {}, headers: { "x-should-retry": "false" } },
      ]),
    });
    await expect(settle(store.getConversation("conv_1"))).rejects.toThrow(
      /503/,
    );
    expect(calls).toHaveLength(1);

    const again: Call[] = [];
    const hinted = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(again, [
        { status: 400, json: {}, headers: { "x-should-retry": "true" } },
        { status: 200, json: CONV },
      ]),
    });
    expect((await settle(hinted.getConversation("conv_1")))?.id).toBe("conv_1");
    expect(again).toHaveLength(2);
  });

  it("waits as long as retry-after-ms asks before the next attempt", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        { status: 429, json: {}, headers: { "retry-after-ms": "2500" } },
        { status: 200, json: CONV },
      ]),
    });

    const pending = store.getConversation("conv_1");
    await vi.advanceTimersByTimeAsync(2499);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls).toHaveLength(2);
    expect((await pending)?.id).toBe("conv_1");
  });

  it("retries a read whose connection dropped", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        { reject: networkError("ECONNRESET") },
        { status: 200, json: CONV },
      ]),
    });

    expect((await settle(store.getConversation("conv_1")))?.id).toBe("conv_1");
    expect(calls).toHaveLength(2);
  });

  it("never re-sends an append whose outcome is unknown", async () => {
    // A 5xx from a gateway or a connection lost mid-request may still have
    // landed; a second send would duplicate the items in the transcript.
    for (const step of [
      { status: 504, text: CLOUDFLARE_504 },
      { status: 500, json: { error: { message: "server error" } } },
      { reject: networkError("ECONNRESET") },
    ]) {
      const calls: Call[] = [];
      const store = new OpenAIConversationStore({
        apiKey: "sk-test",
        fetch: scripted(calls, [step]),
      });

      await expect(
        settle(store.appendItems("conv_1", [msg("hello")])),
      ).rejects.toThrow();
      expect(calls).toHaveLength(1);
    }
  });

  it("re-sends an append the server refused before it could land", async () => {
    for (const refusal of [
      { status: 429, json: { error: { message: "rate limited" } } },
      { status: 500, json: {}, headers: { "x-should-retry": "true" } },
      { reject: networkError("ECONNREFUSED") },
    ]) {
      const calls: Call[] = [];
      const store = new OpenAIConversationStore({
        apiKey: "sk-test",
        fetch: scripted(calls, [refusal, { status: 200, json: { data: [] } }]),
      });

      await settle(store.appendItems("conv_1", [msg("hello")]));

      expect(calls).toHaveLength(2);
      expect(calls[1].body).toEqual(calls[0].body);
    }
  });

  it("retries creating a conversation and returns the id it finally got", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [
        { status: 503, text: "" },
        { status: 200, json: { ...CONV, id: "conv_2" } },
      ]),
    });

    expect((await settle(store.createConversation({}))).id).toBe("conv_2");
    expect(calls.map((c) => c.method)).toEqual(["POST", "POST"]);
  });

  it("sends each request once with maxRetries: 0", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      maxRetries: 0,
      fetch: scripted(calls, [{ status: 504, text: CLOUDFLARE_504 }]),
    });

    const err = await settle(store.getConversation("conv_1")).catch((e) => e);

    expect(err).toBeInstanceOf(OpenAIStoreError);
    expect(err.attempts).toBe(1);
    expect(err.message).toBe(
      "OpenAI conversations API error 504: api.openai.com | 504: Gateway time-out (HTML error page, Cloudflare Ray ID a43247a5be46f282) [GET /conversations/conv_1]",
    );
    expect(calls).toHaveLength(1);
  });

  it("stops retrying the moment the caller aborts", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      fetch: scripted(calls, [{ status: 504, text: CLOUDFLARE_504 }]),
    });
    const ac = new AbortController();

    const pending = store.getConversation("conv_1", ac.signal).catch((e) => e);
    await vi.advanceTimersByTimeAsync(0); // the first attempt fails; backoff starts
    ac.abort(new Error("caller gave up"));
    const err = await pending;

    expect(err.message).toBe("caller gave up");
    expect(calls).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0); // no backoff or timeout left behind
  });

  it("abandons a read that never starts answering, and retries it", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      timeoutMs: 1000,
      fetch: scripted(calls, [{ hang: true }, { status: 200, json: CONV }]),
    });

    const pending = store.getConversation("conv_1");
    await vi.advanceTimersByTimeAsync(999);
    expect(calls).toHaveLength(1); // still waiting on the first attempt
    await vi.runAllTimersAsync();

    expect((await pending)?.id).toBe("conv_1");
    expect(calls).toHaveLength(2);
  });

  it("reports a read that never answered as a timeout, not an abort", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      timeoutMs: 1000,
      fetch: scripted(calls, [{ hang: true }, { hang: true }, { hang: true }]),
    });

    const err = await settle(store.getConversation("conv_1")).catch((e) => e);

    expect(err).toBeInstanceOf(OpenAIStoreTimeoutError);
    expect(err.attempts).toBe(3);
    expect(err.message).toBe(
      "OpenAI conversations API did not respond within 1000 ms [GET /conversations/conv_1, 3 attempts]",
    );
    // Callers that treat "abort" as "the caller ran out of time" must not
    // mistake an upstream stall for their own deadline.
    expect(err.name).not.toBe("AbortError");
    expect(err.message).not.toMatch(/abort/i);
    expect(calls).toHaveLength(3);
  });

  it("lets an append run past the timeout rather than abandon a write that may land", async () => {
    const calls: Call[] = [];
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      timeoutMs: 1000,
      fetch: scripted(calls, [
        { delayMs: 5000, status: 200, json: { data: [] } },
      ]),
    });

    await settle(store.appendItems("conv_1", [msg("hello")]));

    expect(calls).toHaveLength(1);
  });

  it("keeps the clock off a response that has started — only the wait for headers counts", async () => {
    // A large page may take a while to download once it has started. The body
    // is tied to the request's signal, as with a real fetch, so a timer left
    // running would cut the download off.
    const store = new OpenAIConversationStore({
      apiKey: "sk-test",
      timeoutMs: 1000,
      fetch: (async (_url: any, init: any) => ({
        ok: true,
        status: 200,
        headers: new Headers(),
        text: () =>
          new Promise((resolve, reject) => {
            setTimeout(() => resolve(JSON.stringify(CONV)), 5000);
            init.signal.addEventListener(
              "abort",
              () => reject(init.signal.reason),
              { once: true },
            );
          }),
      })) as unknown as typeof fetch,
    });

    expect((await settle(store.getConversation("conv_1")))?.id).toBe("conv_1");
  });

  it("rejects retry settings it cannot honor, when the store is built", () => {
    for (const bad of [
      { maxRetries: -1 },
      { maxRetries: 1.5 },
      { maxRetries: Number.NaN },
      { timeoutMs: -1 },
      { timeoutMs: Number.POSITIVE_INFINITY },
      { timeoutMs: 2 ** 31 },
    ]) {
      expect(
        () => new OpenAIConversationStore({ apiKey: "sk-test", ...bad }),
      ).toThrow(/maxRetries|timeoutMs/);
      expect(() =>
        createStoreForClient("openAI", bad, { apiKey: "sk-test" }),
      ).toThrow(/maxRetries|timeoutMs/);
    }
  });

  it("passes store_config retry settings through to the store", async () => {
    const calls: Call[] = [];
    const store = createStoreForClient(
      "openAI",
      {
        maxRetries: 0,
        fetch: scripted(calls, [{ status: 504, text: CLOUDFLARE_504 }]),
      },
      { apiKey: "sk-test" },
    );

    await expect(settle(store.getConversation("conv_1"))).rejects.toThrow(
      /504/,
    );
    expect(calls).toHaveLength(1);
  });
});
