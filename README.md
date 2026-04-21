# responses-to-completions

A drop-in proxy that exposes the **OpenAI Responses API** on top of any
OpenAI-compatible `/v1/chat/completions` backend (vLLM, Ollama, llama.cpp,
TGI, LiteLLM, Together, Groq, OpenRouter, …).

Point your client at this server instead of `api.openai.com` and it will:

- Accept `POST /v1/responses` requests in Responses-API shape.
- Translate them to chat-completions calls against your backend.
- Persist **conversations and responses** to a pluggable **Store** (local file or S3).
- Execute **MCP tools** server-side on the model's behalf.
- Stream back proper **Responses-API SSE events** (`response.created`, `response.output_text.delta`, `response.output_item.done`, `response.completed`, …).

## Why

The Responses API unifies conversation state, tool execution, and streaming
into a single server-managed primitive. Open-source and self-hosted models
only speak the older Chat Completions protocol. This package bridges the gap
so you can migrate by changing one URL.

## Install

```bash
npm i responses-to-completions
```

## Quick start — run as a server

```bash
BACKEND_BASE_URL=http://localhost:8000/v1 \
STORE_LOCAL_ROOT=./.data \
npx responses-to-completions
```

Point the OpenAI SDK at it:

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8787/v1",
  apiKey: "not-used-but-required",
});

const resp = await client.responses.create({
  model: "qwen2.5-coder:32b",
  input: "Write a haiku about TypeScript.",
});
console.log(resp.output_text);
```

Streaming:

```ts
const stream = await client.responses.create({
  model: "qwen2.5-coder:32b",
  input: "Tell me a story",
  stream: true,
});
for await (const event of stream) {
  if (event.type === "response.output_text.delta") process.stdout.write(event.delta);
}
```

## Backends

### OpenAI-compatible (default)

Works with any server implementing `POST /v1/chat/completions` — including
Ollama's own `/v1/chat/completions` OpenAI-compat endpoint (port 11434).

```bash
BACKEND=openai-compat
BACKEND_BASE_URL=http://localhost:8000/v1
BACKEND_API_KEY=optional-bearer
```

### Ollama (native `/api/chat`)

Normalizes Ollama-specific differences (NDJSON streaming, tool arguments as
objects, no `developer` role):

```bash
BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
```

Both adapters accept `FORCE_MODEL=...` to override the client's requested
model on every upstream call — useful when your backend serves only one model.

## Stores (conversation persistence)

State is persisted to a **Store**, a pluggable interface (`Store`) with two
built-in implementations.

### Local file

```bash
STORE=local
STORE_LOCAL_ROOT=./.data
```

Writes one JSON file per conversation and per response under:

```
<root>/conversations/<id>.json
<root>/responses/<id>.json
```

### S3

```bash
STORE=s3
STORE_S3_BUCKET=my-bucket
STORE_S3_PREFIX=prod/responses
STORE_S3_REGION=us-east-1
# Standard AWS SDK credentials resolution applies.
```

### Custom stores

Implement the `Store` interface and pass it when constructing the server
programmatically:

```ts
import { createServer, OpenAICompatAdapter, type Store } from "responses-to-completions";

const store: Store = /* your Mongo / Postgres / Redis / DynamoDB impl */;
const app = createServer({
  backend: new OpenAICompatAdapter({ baseUrl: "https://api.openai.com/v1", apiKey: process.env.OPENAI_API_KEY }),
  store,
});
app.listen(8787);
```

## Conversations API

All conversation endpoints are implemented:

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/conversations` | Create a conversation with optional initial `items` |
| `GET` | `/v1/conversations/:id` | Retrieve metadata |
| `POST` | `/v1/conversations/:id` | Update metadata |
| `DELETE` | `/v1/conversations/:id` | Delete a conversation |
| `GET` | `/v1/conversations/:id/items` | List items (`limit`, `after`, `order`) |
| `POST` | `/v1/conversations/:id/items` | Append items |
| `GET` | `/v1/conversations/:id/items/:itemId` | Get a single item |
| `DELETE` | `/v1/conversations/:id/items/:itemId` | Delete an item |

Items are stored canonically as typed Responses-API items
(`message`, `function_call`, `function_call_output`, `mcp_list_tools`,
`mcp_call`, `mcp_approval_request`, `reasoning`) and rehydrated into a
chat-completions `messages[]` view before each upstream call.

Pass a conversation to `/v1/responses` to continue it:

```ts
await client.responses.create({
  model: "llama3.1:70b",
  conversation: "conv_abc123",     // server auto-creates if missing
  input: "What did I just ask?",
});
```

Or chain with `previous_response_id`, same as OpenAI.

## MCP tools

Remote MCP servers are supported via the standard Responses-API `tools`
entry (`type: "mcp"`). The proxy connects to each MCP server at request
time, lists its tools, exposes them to the backend model as function tools,
and executes any tool calls server-side — mirroring OpenAI's behavior.

```ts
await client.responses.create({
  model: "llama3.1:70b",
  tools: [
    {
      type: "mcp",
      server_label: "docs",
      server_url: "https://docs.example.com/mcp",
      authorization: process.env.DOCS_MCP_TOKEN,
      allowed_tools: ["search_docs", "read_page"],
      require_approval: "never",
    },
  ],
  input: "Find the section on rate limits.",
});
```

The response surface includes:

- `mcp_list_tools` items — the set of tools discovered on each server
- `mcp_call` items — each executed tool call with its output
- `mcp_approval_request` items — when `require_approval` demands one

To approve a paused call, send a follow-up request with an
`mcp_approval_response` input item and `previous_response_id`.

### `require_approval`

Supports the full OpenAI shape:

- `"never"` — execute all tools immediately (default in this package).
- `"always"` — emit an `mcp_approval_request` for every call.
- `{ always: { tool_names: [...] }, never: { tool_names: [...] } }` — per-tool.

### Not yet supported

- OpenAI **connectors** (`connector_id`) — only raw `server_url` MCP servers.
- Built-in tools `web_search`, `file_search`, `code_interpreter`,
  `computer_use`, `image_generation` — out of scope for v1. Requests including
  them will fail at the backend since we forward them as-is without translation.

## Library usage

If you'd rather wire routes into your own Express app:

```ts
import express from "express";
import {
  mountRoutes,
  OpenAICompatAdapter,
  LocalFileStore,
} from "responses-to-completions";

const app = express();
app.use(express.json());

mountRoutes(app, {
  backend: new OpenAICompatAdapter({ baseUrl: "http://localhost:8000/v1" }),
  store: new LocalFileStore("./.data"),
});

app.listen(8787);
```

Or skip routing entirely and drive the translator yourself:

```ts
import { AgentLoop, OpenAICompatAdapter } from "responses-to-completions";

const agent = new AgentLoop({ backend: new OpenAICompatAdapter({ baseUrl: "..." }) });
const result = await agent.run({
  request: { model: "gpt-4o-mini", input: "hi" },
  history: [],
});
```

## Configuration reference

| Env var | Default | Description |
| --- | --- | --- |
| `PORT` | `8787` | HTTP port |
| `HOST` | `0.0.0.0` | Bind address |
| `BACKEND` | `openai-compat` | `openai-compat` or `ollama` |
| `BACKEND_BASE_URL` | — | Base URL for openai-compat (e.g. `https://api.openai.com/v1`) |
| `BACKEND_API_KEY` | — | Bearer token for the backend |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama host |
| `FORCE_MODEL` | — | Override client's model on every upstream call |
| `STORE` | `local` | `local` or `s3` |
| `STORE_LOCAL_ROOT` | `./.data` | Directory for local store |
| `STORE_S3_BUCKET` | — | S3 bucket for S3 store |
| `STORE_S3_PREFIX` | — | S3 key prefix |
| `STORE_S3_REGION` | — | S3 region (falls back to AWS SDK default chain) |
| `MAX_ITERATIONS` | `10` | Max agent-loop turns per request |

## Design notes

- **Items are the source of truth.** Conversations are persisted as an
  ordered array of typed items. Each request rehydrates items →
  chat-completions messages, runs the loop, appends new items.
- **Agent loop owns MCP execution.** Non-MCP function tools are passed
  through to the client (`function_call` items), same as OpenAI.
- **Streaming is synthesized.** Each `chat.completion.chunk` is mapped
  into the correct Responses-API event sequence; multi-turn tool loops
  serialize per iteration (stream deltas → execute tools → stream next).
- **No hidden writes.** Setting `store: false` on a request skips persistence.

## License

MIT
