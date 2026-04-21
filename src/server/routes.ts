import express, { type Express, type Request, type Response, type NextFunction } from "express";
import { AgentLoop } from "../agent-loop.js";
import type { BackendAdapter } from "../backend/adapter.js";
import type { Store, ConversationItem } from "../store/store.js";
import type {
  ConversationObject,
  CreateResponseRequest,
  InputItem,
  ResponseObject,
} from "../types/responses.js";
import { genConvId, genResponseId, now } from "../util/ids.js";
import { SseWriter } from "./sse-writer.js";

export interface RouteDeps {
  store: Store;
  backend: BackendAdapter;
  maxIterations?: number;
}

/**
 * Mounts `/v1/responses`, `/v1/responses/:id` and `/v1/conversations/*`
 * routes onto a freshly-created Express app. Users who want to mount into
 * an existing app should call `mountRoutes(app, deps)` directly.
 */
export function createServer(deps: RouteDeps): Express {
  const app = express();
  app.use(express.json({ limit: "50mb" }));
  mountRoutes(app, deps);
  app.use(errorHandler);
  return app;
}

export function mountRoutes(app: Express, deps: RouteDeps): void {
  const agent = new AgentLoop({ backend: deps.backend, maxIterations: deps.maxIterations });

  // --- /v1/responses ---
  app.post("/v1/responses", async (req, res, next) => {
    try {
      await handleCreateResponse(req, res, deps, agent);
    } catch (err) {
      next(err);
    }
  });

  app.get("/v1/responses/:id", async (req, res, next) => {
    try {
      const resp = await deps.store.getResponse(req.params.id);
      if (!resp) return res.status(404).json(errorBody("not_found", "Response not found"));
      res.json(resp);
    } catch (err) {
      next(err);
    }
  });

  app.delete("/v1/responses/:id", async (req, res, next) => {
    try {
      const { id, deleted } = await deps.store.deleteResponse(req.params.id);
      res.json({ id, object: "response.deleted", deleted });
    } catch (err) {
      next(err);
    }
  });

  app.get("/v1/responses/:id/input_items", async (req, res, next) => {
    try {
      const resp = await deps.store.getResponse(req.params.id);
      if (!resp) return res.status(404).json(errorBody("not_found", "Response not found"));
      // Our ResponseObject doesn't separately store input_items; we use the
      // conversation's item history up to (and including) this response. For
      // simplicity and until we persist per-response input snapshots, return
      // an empty list rather than lying about the data.
      res.json({ object: "list", data: [], first_id: null, last_id: null, has_more: false });
    } catch (err) {
      next(err);
    }
  });

  // --- /v1/conversations ---
  app.post("/v1/conversations", async (req, res, next) => {
    try {
      const { metadata, items } = req.body ?? {};
      const convo = await deps.store.createConversation({
        id: genConvId(),
        metadata: metadata ?? null,
        items: items ?? [],
      });
      res.json(convo);
    } catch (err) {
      next(err);
    }
  });

  app.get("/v1/conversations/:id", async (req, res, next) => {
    try {
      const convo = await deps.store.getConversation(req.params.id);
      if (!convo) return res.status(404).json(errorBody("not_found", "Conversation not found"));
      res.json(convo);
    } catch (err) {
      next(err);
    }
  });

  app.post("/v1/conversations/:id", async (req, res, next) => {
    try {
      const updated = await deps.store.updateConversation(req.params.id, {
        metadata: req.body?.metadata ?? null,
      });
      if (!updated) return res.status(404).json(errorBody("not_found", "Conversation not found"));
      res.json(updated);
    } catch (err) {
      next(err);
    }
  });

  app.delete("/v1/conversations/:id", async (req, res, next) => {
    try {
      const { id, deleted } = await deps.store.deleteConversation(req.params.id);
      res.json({ id, object: "conversation.deleted", deleted });
    } catch (err) {
      next(err);
    }
  });

  app.get("/v1/conversations/:id/items", async (req, res, next) => {
    try {
      const limit = parseInt(String(req.query.limit ?? "100"), 10);
      const after = req.query.after ? String(req.query.after) : undefined;
      const order = (req.query.order === "desc" ? "desc" : "asc") as "asc" | "desc";
      const { items, hasMore } = await deps.store.listItems(req.params.id, {
        limit,
        after,
        order,
      });
      res.json({
        object: "list",
        data: items,
        first_id: items[0] ? getItemId(items[0]) : null,
        last_id: items[items.length - 1] ? getItemId(items[items.length - 1]) : null,
        has_more: hasMore,
      });
    } catch (err) {
      next(err);
    }
  });

  app.post("/v1/conversations/:id/items", async (req, res, next) => {
    try {
      const items = (req.body?.items ?? []) as ConversationItem[];
      await deps.store.appendItems(req.params.id, items);
      const { items: all } = await deps.store.listItems(req.params.id);
      res.json({
        object: "list",
        data: all,
        first_id: all[0] ? getItemId(all[0]) : null,
        last_id: all[all.length - 1] ? getItemId(all[all.length - 1]) : null,
        has_more: false,
      });
    } catch (err) {
      next(err);
    }
  });

  app.get("/v1/conversations/:id/items/:itemId", async (req, res, next) => {
    try {
      const item = await deps.store.getItem(req.params.id, req.params.itemId);
      if (!item) return res.status(404).json(errorBody("not_found", "Item not found"));
      res.json(item);
    } catch (err) {
      next(err);
    }
  });

  app.delete("/v1/conversations/:id/items/:itemId", async (req, res, next) => {
    try {
      const r = await deps.store.deleteItem(req.params.id, req.params.itemId);
      res.json({ id: r.id, object: "conversation.item.deleted", deleted: r.deleted });
    } catch (err) {
      next(err);
    }
  });

  // Health
  app.get("/healthz", (_req, res) => res.json({ ok: true }));
}

async function handleCreateResponse(
  req: Request,
  res: Response,
  deps: RouteDeps,
  agent: AgentLoop,
): Promise<void> {
  const body = req.body as CreateResponseRequest;
  if (!body?.model) {
    res.status(400).json(errorBody("invalid_request_error", "`model` is required"));
    return;
  }
  if (body.input === undefined && !body.previous_response_id) {
    res.status(400).json(errorBody("invalid_request_error", "`input` or `previous_response_id` is required"));
    return;
  }

  // Resolve history from conversation / previous_response_id.
  const { history, conversationId } = await resolveHistory(deps.store, body);

  const stream = Boolean(body.stream);
  const signal = abortSignalFromReq(req);

  const responseId = genResponseId();
  const initialResp = buildInitialResponse(responseId, body, conversationId);

  const newInputItems = normalizeNewInputItems(body.input);

  if (stream) {
    const writer = new SseWriter(res);
    req.on("close", () => writer.close());

    let seq = 0;
    writer.send("response.created", {
      type: "response.created",
      sequence_number: seq++,
      response: initialResp,
    });
    writer.send("response.in_progress", {
      type: "response.in_progress",
      sequence_number: seq++,
      response: initialResp,
    });

    try {
      const gen = agent.stream({ request: body, history, signal });
      let result: { items: Awaited<ReturnType<typeof agent.run>>["items"]; usage: Awaited<ReturnType<typeof agent.run>>["usage"] } = { items: [], usage: null };
      while (true) {
        const r = await gen.next();
        if (r.done) {
          result = r.value;
          break;
        }
        const ev = r.value;
        // Re-tag sequence_number into the outer counter.
        const tagged = { ...ev, sequence_number: seq++ };
        writer.send(ev.type, tagged);
      }

      const finalResp: ResponseObject = {
        ...initialResp,
        status: "completed",
        output: result.items,
        output_text: aggregateText(result.items),
        usage: result.usage,
      };
      if (body.store !== false) {
        if (conversationId) {
          await deps.store.appendItems(conversationId, [...newInputItems, ...result.items]);
        }
        await deps.store.saveResponse(finalResp);
      }
      writer.send("response.completed", {
        type: "response.completed",
        sequence_number: seq++,
        response: finalResp,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const failedResp: ResponseObject = {
        ...initialResp,
        status: "failed",
        error: { code: "internal_error", message },
      };
      try {
        writer.send("response.failed", {
          type: "response.failed",
          sequence_number: seq++,
          response: failedResp,
        });
      } catch {}
      if (body.store !== false) {
        await deps.store.saveResponse(failedResp).catch(() => {});
      }
    } finally {
      writer.close();
    }
    return;
  }

  // Non-streaming
  try {
    const result = await agent.run({ request: body, history, signal });
    const finalResp: ResponseObject = {
      ...initialResp,
      status: "completed",
      output: result.items,
      output_text: aggregateText(result.items),
      usage: result.usage,
    };
    if (body.store !== false) {
      if (conversationId) {
        await deps.store.appendItems(conversationId, [...newInputItems, ...result.items]);
      }
      await deps.store.saveResponse(finalResp);
    }
    res.json(finalResp);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const failedResp: ResponseObject = {
      ...initialResp,
      status: "failed",
      error: { code: "internal_error", message },
    };
    if (body.store !== false) {
      await deps.store.saveResponse(failedResp).catch(() => {});
    }
    res.status(500).json(failedResp);
  }
}

// ---- helpers ----

async function resolveHistory(
  store: Store,
  body: CreateResponseRequest,
): Promise<{ history: ConversationItem[]; conversationId: string | null }> {
  // Priority: explicit conversation > previous_response_id > none.
  if (body.conversation) {
    const convId = typeof body.conversation === "string" ? body.conversation : body.conversation.id;
    // Auto-create conversation if it doesn't exist so clients can use client-generated IDs.
    const existing = await store.getConversation(convId);
    if (!existing) {
      await store.createConversation({ id: convId });
      return { history: [], conversationId: convId };
    }
    const { items } = await store.listItems(convId);
    return { history: items, conversationId: convId };
  }

  if (body.previous_response_id) {
    const prev = await store.getResponse(body.previous_response_id);
    if (!prev) {
      throw new HttpError(400, `previous_response_id not found: ${body.previous_response_id}`);
    }
    // If the previous response was associated with a conversation, continue it.
    if (prev.conversation?.id) {
      const { items } = await store.listItems(prev.conversation.id);
      return { history: items, conversationId: prev.conversation.id };
    }
    // Otherwise just hand the prev response's outputs as history (no persistence).
    return { history: prev.output as ConversationItem[], conversationId: null };
  }

  return { history: [], conversationId: null };
}

function normalizeNewInputItems(input: string | InputItem[] | undefined): ConversationItem[] {
  if (input === undefined) return [];
  if (typeof input === "string") {
    return [{ type: "message", role: "user", content: input }] as ConversationItem[];
  }
  return input as ConversationItem[];
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
    temperature: body.temperature ?? null,
    tool_choice: body.tool_choice ?? "auto",
    tools: body.tools ?? [],
    top_p: body.top_p ?? null,
    usage: null,
    user: body.user ?? null,
    metadata: body.metadata ?? null,
  };
}

function aggregateText(items: ConversationItem[]): string {
  let out = "";
  for (const it of items) {
    if ((it as { type?: string }).type === "message") {
      const parts = (it as { content?: Array<{ type: string; text?: string }> }).content ?? [];
      for (const p of parts) {
        if (p.type === "output_text" && p.text) out += p.text;
      }
    }
  }
  return out;
}

function getItemId(item: ConversationItem): string | null {
  return ((item as { id?: string }).id ?? (item as { call_id?: string }).call_id) ?? null;
}

class HttpError extends Error {
  constructor(public status: number, msg: string) {
    super(msg);
  }
}

function errorBody(code: string, message: string) {
  return { error: { code, message, type: "invalid_request_error" } };
}

function errorHandler(err: unknown, _req: Request, res: Response, _next: NextFunction): void {
  if (err instanceof HttpError) {
    res.status(err.status).json(errorBody("invalid_request_error", err.message));
    return;
  }
  const message = err instanceof Error ? err.message : String(err);
  // eslint-disable-next-line no-console
  console.error("[responses-to-completions] Unhandled error:", err);
  res.status(500).json(errorBody("internal_error", message));
}

function abortSignalFromReq(req: Request): AbortSignal {
  const ac = new AbortController();
  req.on("close", () => ac.abort());
  return ac.signal;
}

function _unused(_: ConversationObject) {
  // keep ConversationObject import referenced for type inference consumers
}
