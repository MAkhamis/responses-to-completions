import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand,
  type S3ClientConfig,
} from "@aws-sdk/client-s3";
import type { ConversationObject, ResponseObject } from "../types/responses.js";
import type { ConversationItem, Store } from "./store.js";
import { genConvId, now } from "../util/ids.js";

export interface S3StoreOptions {
  bucket: string;
  prefix?: string;
  client?: S3Client;
  clientConfig?: S3ClientConfig;
}

/**
 * Stores conversations and responses as JSON objects in S3.
 *
 * Keys:
 *   <prefix>/conversations/<id>.json
 *   <prefix>/responses/<id>.json
 *
 * S3 offers read-after-write consistency, so naive append is OK for low
 * contention. For high-contention producers, put a queue in front or use
 * S3 object locks / DynamoDB — out of scope here.
 *
 * Writes are serialized per-key by an in-process lock, same as the local
 * store; this covers the common single-proxy deployment.
 */
export class S3Store implements Store {
  private client: S3Client;
  private bucket: string;
  private prefix: string;
  private locks = new Map<string, Promise<void>>();

  constructor(opts: S3StoreOptions) {
    this.client = opts.client ?? new S3Client(opts.clientConfig ?? {});
    this.bucket = opts.bucket;
    this.prefix = (opts.prefix ?? "").replace(/^\/+|\/+$/g, "");
  }

  private key(kind: "conversations" | "responses", id: string) {
    const safe = encodeURIComponent(id);
    return [this.prefix, kind, `${safe}.json`].filter(Boolean).join("/");
  }

  private async withLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.locks.get(key) ?? Promise.resolve();
    let release!: () => void;
    const next = new Promise<void>((r) => (release = r));
    const mine = prev.then(() => next);
    this.locks.set(key, mine);
    await prev;
    try {
      return await fn();
    } finally {
      release();
      if (this.locks.get(key) === mine) this.locks.delete(key);
    }
  }

  private async getJson<T>(
    key: string,
    signal?: AbortSignal,
  ): Promise<T | null> {
    try {
      const resp = await this.client.send(
        new GetObjectCommand({ Bucket: this.bucket, Key: key }),
        { abortSignal: signal },
      );
      const body = await resp.Body?.transformToString("utf-8");
      if (!body) return null;
      return JSON.parse(body) as T;
    } catch (e) {
      const name =
        (e as { name?: string; Code?: string }).name ??
        (e as { Code?: string }).Code;
      if (name === "NoSuchKey" || name === "NotFound") return null;
      throw e;
    }
  }

  private async putJson(key: string, data: unknown, signal?: AbortSignal) {
    await this.client.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: JSON.stringify(data),
        ContentType: "application/json",
      }),
      { abortSignal: signal },
    );
  }

  private async del(key: string): Promise<boolean> {
    try {
      await this.client.send(
        new DeleteObjectCommand({ Bucket: this.bucket, Key: key }),
      );
      return true;
    } catch (e) {
      const name = (e as { name?: string }).name;
      if (name === "NoSuchKey" || name === "NotFound") return false;
      throw e;
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
    const id = input.id ?? genConvId();
    const convo: ConversationObject & { items: ConversationItem[] } = {
      id,
      object: "conversation",
      created_at: now(),
      metadata: input.metadata ?? null,
      items: input.items ?? [],
    };
    await this.putJson(this.key("conversations", id), convo, signal);
    const { items: _omit, ...meta } = convo;
    return meta;
  }

  async getConversation(
    id: string,
    signal?: AbortSignal,
  ): Promise<ConversationObject | null> {
    const data = await this.getJson<
      ConversationObject & { items: ConversationItem[] }
    >(this.key("conversations", id), signal);
    if (!data) return null;
    const { items: _omit, ...meta } = data;
    return meta;
  }

  async updateConversation(
    id: string,
    patch: { metadata?: Record<string, string> | null },
  ): Promise<ConversationObject | null> {
    return this.withLock(`conv:${id}`, async () => {
      const data = await this.getJson<
        ConversationObject & { items: ConversationItem[] }
      >(this.key("conversations", id));
      if (!data) return null;
      if (patch.metadata !== undefined) data.metadata = patch.metadata;
      await this.putJson(this.key("conversations", id), data);
      const { items: _omit, ...meta } = data;
      return meta;
    });
  }

  async deleteConversation(
    id: string,
  ): Promise<{ id: string; deleted: boolean }> {
    const deleted = await this.del(this.key("conversations", id));
    return { id, deleted };
  }

  // ---- items ----
  async appendItems(
    conversationId: string,
    items: ConversationItem[],
    signal?: AbortSignal,
  ): Promise<void> {
    if (items.length === 0) return;
    await this.withLock(`conv:${conversationId}`, async () => {
      const data = await this.getJson<
        ConversationObject & { items: ConversationItem[] }
      >(this.key("conversations", conversationId), signal);
      if (!data) throw new Error(`Conversation not found: ${conversationId}`);
      data.items.push(...items);
      await this.putJson(
        this.key("conversations", conversationId),
        data,
        signal,
      );
    });
  }

  async listItems(
    conversationId: string,
    opts?: { limit?: number; after?: string; order?: "asc" | "desc" },
    signal?: AbortSignal,
  ): Promise<{ items: ConversationItem[]; hasMore: boolean }> {
    const data = await this.getJson<
      ConversationObject & { items: ConversationItem[] }
    >(this.key("conversations", conversationId), signal);
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
    const data = await this.getJson<
      ConversationObject & { items: ConversationItem[] }
    >(this.key("conversations", conversationId));
    if (!data) return null;
    return data.items.find((it) => getItemId(it) === itemId) ?? null;
  }

  async deleteItem(
    conversationId: string,
    itemId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.withLock(`conv:${conversationId}`, async () => {
      const data = await this.getJson<
        ConversationObject & { items: ConversationItem[] }
      >(this.key("conversations", conversationId));
      if (!data) return { id: itemId, deleted: false };
      const before = data.items.length;
      data.items = data.items.filter((it) => getItemId(it) !== itemId);
      if (data.items.length === before) return { id: itemId, deleted: false };
      await this.putJson(this.key("conversations", conversationId), data);
      return { id: itemId, deleted: true };
    });
  }

  // ---- responses ----
  async saveResponse(
    resp: ResponseObject,
    signal?: AbortSignal,
  ): Promise<void> {
    await this.putJson(this.key("responses", resp.id), resp, signal);
  }

  async getResponse(
    id: string,
    signal?: AbortSignal,
  ): Promise<ResponseObject | null> {
    return this.getJson<ResponseObject>(this.key("responses", id), signal);
  }

  async deleteResponse(id: string): Promise<{ id: string; deleted: boolean }> {
    const deleted = await this.del(this.key("responses", id));
    return { id, deleted };
  }
}

function getItemId(it: ConversationItem): string | undefined {
  return (it as { id?: string }).id ?? (it as { call_id?: string }).call_id;
}
