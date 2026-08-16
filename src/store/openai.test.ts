import { describe, expect, it } from "vitest";
import { OpenAIConversationStore } from "./openai.js";
import type { ConversationItem } from "./store.js";

type Call = { url: string; method: string; body: any };

const mockFetch = (
  calls: Call[],
  routes: (url: string, method: string, body: any) => { status?: number; json: any },
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

    await expect(
      store.createConversation({ id: "conv_mine" }),
    ).rejects.toThrow(/OpenAI assigns conversation ids/);
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
        json: { id: "conv_1", object: "conversation", created_at: 1, metadata: null },
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
