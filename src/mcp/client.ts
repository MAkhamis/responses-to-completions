import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type {
  McpToolDef,
  RequireApproval,
} from "../types/responses.js";

export interface McpToolInfo {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  annotations?: Record<string, unknown>;
}

/**
 * Thin wrapper around the official MCP TypeScript SDK. Connects to a remote
 * MCP server over Streamable HTTP, lists its tools, and executes calls.
 *
 * One instance per (server_label, server_url) per request — cheap to build,
 * connections are not pooled across requests yet.
 */
export class McpConnection {
  private client: Client;
  private connected = false;

  constructor(
    public readonly def: McpToolDef,
    client?: Client,
  ) {
    this.client =
      client ??
      new Client(
        { name: "responses-to-completions", version: "0.1.0" },
        { capabilities: {} },
      );
  }

  async connect(): Promise<void> {
    if (this.connected) return;
    if (!this.def.server_url) {
      throw new Error(
        `MCP tool "${this.def.server_label}" requires server_url (connector_id is not supported in v1)`,
      );
    }
    const headers: Record<string, string> = { ...(this.def.headers ?? {}) };
    if (this.def.authorization) {
      headers["authorization"] = `Bearer ${this.def.authorization}`;
    }
    const transport = new StreamableHTTPClientTransport(new URL(this.def.server_url), {
      requestInit: { headers },
    });
    await this.client.connect(transport);
    this.connected = true;
  }

  async listTools(): Promise<McpToolInfo[]> {
    await this.connect();
    const result = await this.client.listTools();
    const tools = (result.tools ?? []) as Array<{
      name: string;
      description?: string;
      inputSchema?: Record<string, unknown>;
      annotations?: Record<string, unknown>;
    }>;
    const allowed = allowedToolFilter(this.def.allowed_tools);
    return tools
      .filter((t) => allowed(t.name))
      .map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema ?? { type: "object", properties: {} },
        ...(t.annotations ? { annotations: t.annotations } : {}),
      }));
  }

  async call(
    name: string,
    args: Record<string, unknown>,
  ): Promise<{ output: string; isError: boolean }> {
    await this.connect();
    const res = await this.client.callTool({ name, arguments: args });
    const content = Array.isArray(res.content) ? res.content : [];
    const output = content
      .map((c) => {
        const cc = c as { type?: string; text?: string };
        if (cc.type === "text") return cc.text ?? "";
        // image/resource content — represent as a placeholder string
        return JSON.stringify(c);
      })
      .join("");
    return { output, isError: Boolean(res.isError) };
  }

  async close(): Promise<void> {
    if (!this.connected) return;
    try {
      await this.client.close();
    } catch {
      // ignore
    }
  }
}

function allowedToolFilter(
  allowed: McpToolDef["allowed_tools"],
): (name: string) => boolean {
  if (!allowed) return () => true;
  if (Array.isArray(allowed)) {
    const set = new Set(allowed);
    return (n) => set.has(n);
  }
  const names = allowed.tool_names;
  if (!names?.length) return () => true;
  const set = new Set(names);
  return (n) => set.has(n);
}

/** Decides whether a tool call needs user approval per OpenAI's require_approval rules. */
export function needsApproval(
  req: RequireApproval | undefined,
  toolName: string,
): boolean {
  if (req === undefined) return false; // default policy: no approval (server-side executor)
  if (req === "always") return true;
  if (req === "never") return false;
  if (typeof req === "object") {
    if (req.always?.tool_names?.includes(toolName)) return true;
    if (req.never?.tool_names?.includes(toolName)) return false;
    // If only one side is specified, the other side is the default.
    if (req.always && !req.never) return false;
    if (req.never && !req.always) return true;
  }
  return false;
}
