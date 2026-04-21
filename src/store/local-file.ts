import { promises as fs } from "node:fs";
import * as path from "node:path";
import type { ConversationObject, ResponseObject } from "../types/responses.js";
import type { ConversationItem, Store } from "./store.js";
import { genConvId, now } from "../util/ids.js";

/**
 * Stores conversations and responses as JSON files on local disk.
 *
 *   <root>/conversations/<id>.json
 *   <root>/responses/<id>.json
 *
 * A single in-process async lock per key serializes writes so concurrent
 * append-item operations don't clobber each other. Not durable against crashes
 * mid-write (use fs atomic rename helper); not safe across processes.
 */
export class LocalFileStore implements Store {
  private locks = new Map<string, Promise<void>>();

  constructor(private root: string) {}

  private convPath(id: string) {
    return path.join(this.root, "conversations", `${encodeURIComponent(id)}.json`);
  }
  private respPath(id: string) {
    return path.join(this.root, "responses", `${encodeURIComponent(id)}.json`);
  }

  private async ensureDir(file: string) {
    await fs.mkdir(path.dirname(file), { recursive: true });
  }

  private async withLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.locks.get(key) ?? Promise.resolve();
    let release!: () => void;
    const next = new Promise<void>((r) => (release = r));
    this.locks.set(key, prev.then(() => next));
    await prev;
    try {
      return await fn();
    } finally {
      release();
      if (this.locks.get(key) === prev.then(() => next)) this.locks.delete(key);
    }
  }

  private async readJson<T>(file: string): Promise<T | null> {
    try {
      const raw = await fs.readFile(file, "utf8");
      return JSON.parse(raw) as T;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw e;
    }
  }

  private async writeJson(file: string, data: unknown) {
    await this.ensureDir(file);
    const tmp = `${file}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2), "utf8");
    await fs.rename(tmp, file);
  }

  // ---- conversations ----
  async createConversation(input: {
    id?: string;
    metadata?: Record<string, string> | null;
    items?: ConversationItem[];
  }): Promise<ConversationObject> {
    const id = input.id ?? genConvId();
    const convo: ConversationObject & { items: ConversationItem[] } = {
      id,
      object: "conversation",
      created_at: now(),
      metadata: input.metadata ?? null,
      items: input.items ?? [],
    };
    await this.writeJson(this.convPath(id), convo);
    const { items: _omit, ...meta } = convo;
    return meta;
  }

  async getConversation(id: string): Promise<ConversationObject | null> {
    const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
      this.convPath(id),
    );
    if (!data) return null;
    const { items: _omit, ...meta } = data;
    return meta;
  }

  async updateConversation(
    id: string,
    patch: { metadata?: Record<string, string> | null },
  ): Promise<ConversationObject | null> {
    return this.withLock(`conv:${id}`, async () => {
      const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
        this.convPath(id),
      );
      if (!data) return null;
      if (patch.metadata !== undefined) data.metadata = patch.metadata;
      await this.writeJson(this.convPath(id), data);
      const { items: _omit, ...meta } = data;
      return meta;
    });
  }

  async deleteConversation(id: string): Promise<{ id: string; deleted: boolean }> {
    try {
      await fs.unlink(this.convPath(id));
      return { id, deleted: true };
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return { id, deleted: false };
      throw e;
    }
  }

  // ---- items ----
  async appendItems(conversationId: string, items: ConversationItem[]): Promise<void> {
    if (items.length === 0) return;
    await this.withLock(`conv:${conversationId}`, async () => {
      const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
        this.convPath(conversationId),
      );
      if (!data) throw new Error(`Conversation not found: ${conversationId}`);
      data.items.push(...items);
      await this.writeJson(this.convPath(conversationId), data);
    });
  }

  async listItems(
    conversationId: string,
    opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
  ): Promise<{ items: ConversationItem[]; hasMore: boolean }> {
    const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
      this.convPath(conversationId),
    );
    if (!data) return { items: [], hasMore: false };

    let items = data.items.slice();
    if (opts?.order === "desc") items.reverse();
    if (opts?.after) {
      const idx = items.findIndex((it) => getItemId(it) === opts.after);
      if (idx >= 0) items = items.slice(idx + 1);
    }
    const limit = opts?.limit ?? items.length;
    const slice = items.slice(0, limit);
    return { items: slice, hasMore: slice.length < items.length };
  }

  async getItem(
    conversationId: string,
    itemId: string,
  ): Promise<ConversationItem | null> {
    const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
      this.convPath(conversationId),
    );
    if (!data) return null;
    return data.items.find((it) => getItemId(it) === itemId) ?? null;
  }

  async deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.withLock(`conv:${conversationId}`, async () => {
      const data = await this.readJson<ConversationObject & { items: ConversationItem[] }>(
        this.convPath(conversationId),
      );
      if (!data) return { id: itemId, deleted: false };
      const before = data.items.length;
      data.items = data.items.filter((it) => getItemId(it) !== itemId);
      if (data.items.length === before) return { id: itemId, deleted: false };
      await this.writeJson(this.convPath(conversationId), data);
      return { id: itemId, deleted: true };
    });
  }

  // ---- responses ----
  async saveResponse(resp: ResponseObject): Promise<void> {
    await this.writeJson(this.respPath(resp.id), resp);
  }

  async getResponse(id: string): Promise<ResponseObject | null> {
    return this.readJson<ResponseObject>(this.respPath(id));
  }

  async deleteResponse(id: string): Promise<{ id: string; deleted: boolean }> {
    try {
      await fs.unlink(this.respPath(id));
      return { id, deleted: true };
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return { id, deleted: false };
      throw e;
    }
  }
}

function getItemId(it: ConversationItem): string | undefined {
  return (it as { id?: string }).id ?? (it as { call_id?: string }).call_id;
}
