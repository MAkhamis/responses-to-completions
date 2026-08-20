import { describe, expect, it } from "vitest";
import { OpenAIConversationStore } from "./openai.js";
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

  it("forwards an AbortSignal to every fetch, including each page", async () => {
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
    for (const s of signals) expect(s).toBe(ac.signal);
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

  // Fix: OpenAI store getResponse read-through has no test.
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

  // Fix: listItems loop lacks cursor-advance and max-page guards.
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
