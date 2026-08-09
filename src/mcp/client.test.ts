import { createServer } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import { McpConnection } from "./client.js";

const servers: ReturnType<typeof createServer>[] = [];

afterEach(async () => {
  await Promise.all(
    servers
      .splice(0)
      .map(
        (server) =>
          new Promise<void>((resolve, reject) =>
            server.close((error) => (error ? reject(error) : resolve())),
          ),
      ),
  );
});

describe("McpConnection protocol negotiation", () => {
  it("discovers and uses an MCP 2026-07-28 server", async () => {
    const requests: Array<{
      body: Record<string, any>;
      headers: Record<string, string | string[] | undefined>;
    }> = [];
    const server = createServer(async (request, response) => {
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      requests.push({ body, headers: request.headers });

      let result: Record<string, any>;
      switch (body.method) {
        case "server/discover":
          result = {
            resultType: "complete",
            supportedVersions: ["2026-07-28"],
            capabilities: { tools: {} },
            _meta: {
              "io.modelcontextprotocol/serverInfo": {
                name: "test-server",
                version: "1.0.0",
              },
            },
            ttlMs: 60_000,
            cacheScope: "private",
          };
          break;
        case "tools/list":
          result = {
            resultType: "complete",
            tools: [
              {
                name: "echo",
                description: "Echo text",
                inputSchema: {
                  type: "object",
                  properties: { text: { type: "string" } },
                },
              },
            ],
            ttlMs: 60_000,
            cacheScope: "private",
          };
          break;
        case "tools/call":
          result = {
            resultType: "complete",
            content: [
              {
                type: "text",
                text: body.params.arguments.text,
              },
            ],
            isError: false,
          };
          break;
        default:
          throw new Error(`Unexpected MCP method: ${body.method}`);
      }

      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }));
    });
    servers.push(server);
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Expected a TCP test server address");
    }

    const connection = new McpConnection({
      type: "mcp",
      server_label: "test",
      server_url: `http://127.0.0.1:${address.port}/mcp`,
    });

    try {
      await expect(connection.listTools()).resolves.toEqual([
        {
          name: "echo",
          description: "Echo text",
          input_schema: {
            type: "object",
            properties: { text: { type: "string" } },
          },
        },
      ]);
      await expect(connection.call("echo", { text: "hello" })).resolves.toEqual(
        {
          output: "hello",
          isError: false,
        },
      );
    } finally {
      await connection.close();
    }

    expect(requests.map(({ body }) => body.method)).toEqual([
      "server/discover",
      "tools/list",
      "tools/call",
    ]);
    for (const { body, headers } of requests) {
      expect(headers["mcp-protocol-version"]).toBe("2026-07-28");
      expect(headers["mcp-method"]).toBe(body.method);
      expect(body.params._meta["io.modelcontextprotocol/protocolVersion"]).toBe(
        "2026-07-28",
      );
    }
    expect(requests[2].headers["mcp-name"]).toBe("echo");
  });

  it("falls back to the legacy initialize handshake", async () => {
    const methods: string[] = [];
    const server = createServer(async (request, response) => {
      if (request.method !== "POST") {
        response.writeHead(405).end();
        return;
      }

      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      methods.push(body.method);

      if (body.method === "server/discover") {
        response.writeHead(404, { "content-type": "application/json" });
        response.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: body.id,
            error: { code: -32601, message: "Method not found" },
          }),
        );
        return;
      }

      if (body.method === "notifications/initialized") {
        response.writeHead(202).end();
        return;
      }

      const result =
        body.method === "initialize"
          ? {
              protocolVersion: body.params.protocolVersion,
              capabilities: { tools: {} },
              serverInfo: { name: "legacy-server", version: "1.0.0" },
            }
          : {
              tools: [
                {
                  name: "legacy_echo",
                  inputSchema: { type: "object", properties: {} },
                },
              ],
            };
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }));
    });
    servers.push(server);
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Expected a TCP test server address");
    }

    const connection = new McpConnection({
      type: "mcp",
      server_label: "legacy-test",
      server_url: `http://127.0.0.1:${address.port}/mcp`,
    });

    try {
      await expect(connection.listTools()).resolves.toEqual([
        {
          name: "legacy_echo",
          input_schema: { type: "object", properties: {} },
        },
      ]);
    } finally {
      await connection.close();
    }

    expect(methods).toEqual([
      "server/discover",
      "initialize",
      "notifications/initialized",
      "tools/list",
    ]);
  });
});
