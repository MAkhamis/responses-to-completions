# responses-to-completions

An **SDK** that exposes the OpenAI **Responses API** on top of any
OpenAI-compatible `/v1/chat/completions` backend (vLLM, Ollama, llama.cpp,
TGI, LiteLLM, Together, Groq, OpenRouter, …).

Construct a `ResponsesClient`, give it a backend, and call
`client.responses.create(...)` — the client handles conversation state, MCP
tool execution, and streaming-event translation internally.

## Install

```bash
npm i responses-to-completions
```

## Quick start

```ts
import {
  ResponsesClient,
  OpenAICompatAdapter,
  LocalFileStore,
} from "responses-to-completions";

const client = new ResponsesClient({
  backend: new OpenAICompatAdapter({
    baseUrl: "https://api.openai.com/v1",
    apiKey: process.env.OPENAI_API_KEY,
  }),
  // optional — required for conversations and responses.{get,del}
  store: new LocalFileStore("./.data"),
});

const resp = await client.responses.create({
  model: "gpt-4o-mini",
  input: "Write a haiku about TypeScript.",
});
console.log(resp.output_text);
```

### Streaming

`responses.create({ stream: true })` returns a `StreamResponse` — an
async-iterable of `StreamEvent`s plus a `finalResponse()` promise.

```ts
const stream = await client.responses.create({
  model: "gpt-4o-mini",
  input: "Tell me a story.",
  stream: true,
});

for await (const ev of stream) {
  if (ev.type === "response.output_text.delta") {
    process.stdout.write(ev.delta);
  }
}

const finalResp = await stream.finalResponse();
console.log("\nfinal id:", finalResp.id);
```

## Backends

The constructor takes any `BackendAdapter`. Three are built in.

### `OpenAICompatAdapter`

Works with any server implementing `POST /v1/chat/completions` — OpenAI,
vLLM, llama.cpp server, TGI, LiteLLM, Together, Groq, and Ollama's own
OpenAI-compat endpoint on port 11434.

```ts
new OpenAICompatAdapter({
  baseUrl: "http://localhost:8000/v1",
  apiKey: "optional-bearer",
  forceModel: "qwen2.5-coder:32b", // optional override
});
```

### `OllamaAdapter`

Native Ollama `/api/chat`. Normalizes NDJSON streaming, object-shaped tool
arguments, and the missing `developer` role.

```ts
new OllamaAdapter({ host: "http://localhost:11434" });
```

### `OpenRouterAdapter`

OpenRouter with provider-routing preferences.

```ts
new OpenRouterAdapter({
  apiKey: process.env.OPENROUTER_API_KEY!,
  provider: { order: ["Anthropic", "Google"], allow_fallbacks: true },
});
```

### Custom backends

Implement the `BackendAdapter` interface (`complete` + `stream`).

## Embeddings

Call `client.embeddings.create(...)` to produce embeddings. The client
delegates to the configured backend adapter and returns the OpenAI-shaped
`EmbeddingsResponse` regardless of provider.

```ts
import {
  ResponsesClient,
  OpenAICompatAdapter,
} from "responses-to-completions";

const client = new ResponsesClient({
  backend: new OpenAICompatAdapter({
    baseUrl: "https://api.openai.com/v1",
    apiKey: process.env.OPENAI_API_KEY,
  }),
});

const resp = await client.embeddings.create({
  model: "text-embedding-3-large",
  input: "hi",
});
// { object: "list", data: [{ object: "embedding", index, embedding }], model, usage }
```

Works with any built-in adapter — swap in `OllamaAdapter` or
`OpenRouterAdapter` and the call site stays the same:

```ts
import {
  ResponsesClient,
  OllamaAdapter,
  OpenRouterAdapter,
} from "responses-to-completions";

// Ollama — POST /api/embed, translated to the OpenAI shape
const ollamaClient = new ResponsesClient({
  backend: new OllamaAdapter({ host: "http://localhost:11434" }),
});
await ollamaClient.embeddings.create({
  model: "nomic-embed-text",
  input: ["a", "b", "c"],
});

// OpenRouter — routes to whichever provider exposes the model's embeddings
const orClient = new ResponsesClient({
  backend: new OpenRouterAdapter({ apiKey: process.env.OPENROUTER_API_KEY! }),
});
await orClient.embeddings.create({
  model: "openai/text-embedding-3-large",
  input: ["a", "b", "c"],
});
```

Embeddings don't require a `store` — only a `backend`. If the configured
backend doesn't implement `embeddings()`, the call throws.

### Request shape

```ts
interface EmbeddingsRequest {
  model: string;                        // honors `forceModel` on the adapter
  input: string | string[];             // batch by passing an array
  encoding_format?: "float" | "base64"; // default "float"; Ollama ignores
  dimensions?: number;                  // text-embedding-3-* only; others ignore
  user?: string;
}
```

### Adapter-specific notes

- **`OpenAICompatAdapter`** — pass-through to `POST {baseUrl}/embeddings`.
  Reuses the same auth/headers/`forceModel` as `complete()`.
- **`OpenRouterAdapter`** — same path, plus OpenRouter's `provider` routing
  preferences are forwarded. Note that OpenRouter's embeddings coverage is
  narrower than its chat coverage; only models exposing an embeddings
  endpoint will work.
- **`OllamaAdapter`** — calls native `POST /api/embed` and translates the
  response to the OpenAI shape. `encoding_format`, `dimensions`, and `user`
  are silently dropped (Ollama doesn't honor them).

Non-2xx responses throw `BackendError` (same as `complete()`/`respond()`).

## Stores

Persistence is optional. Without a store the client still runs requests
but `conversations.*` and `responses.{get,del}` throw, and
`responses.create` skips writing.

### `LocalFileStore` — for local development / tests

```ts
new LocalFileStore("./.data");
```

Writes one JSON file per artifact:

```
<root>/conversations/<id>.json
<root>/responses/<id>.json
```

### `S3Store` — for production

```ts
new S3Store({
  bucket: "my-bucket",
  prefix: "prod/responses",
  clientConfig: { region: "us-east-1" },
});
```

### Custom stores

Implement the `Store` interface — see `src/store/store.ts`.

## Conversations

Conversation state is owned by the store. The client offers a thin facade:

```ts
const conv = await client.conversations.create({ metadata: { user: "u_42" } });

await client.responses.create({
  model: "gpt-4o-mini",
  conversation: conv.id,
  input: "Remember the secret word 'banana'.",
});
await client.responses.create({
  model: "gpt-4o-mini",
  conversation: conv.id,
  input: "What was the secret word?",
});

const { data } = await client.conversations.items.list(conv.id);
```

Full surface:

| Method | Notes |
| --- | --- |
| `conversations.create({ id?, items?, metadata? })` | `id` is optional; client-generated when omitted |
| `conversations.get(id)` | Returns metadata only (items via `items.list`) |
| `conversations.update(id, { metadata })` | |
| `conversations.del(id)` | |
| `conversations.items.list(id, { limit?, after?, order? })` | Paginated |
| `conversations.items.append(id, items)` | Returns the full item list after append |
| `conversations.items.get(id, itemId)` | |
| `conversations.items.del(id, itemId)` | |

Items are stored canonically as typed Responses-API items (`message`,
`function_call`, `function_call_output`, `mcp_list_tools`, `mcp_call`,
`mcp_approval_request`, `reasoning`) and rehydrated into a chat-completions
`messages[]` view before each upstream call.

You can also continue a conversation by `previous_response_id`, same as
OpenAI.

## MCP tools

Remote MCP servers are supported via the standard Responses-API `tools`
entry (`type: "mcp"`). The client connects to each MCP server at request
time, lists its tools, exposes them to the backend model as function tools,
and executes any tool calls server-side.

```ts
await client.responses.create({
  model: "gpt-4o-mini",
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

- `mcp_list_tools` items — the tools discovered on each server.
- `mcp_call` items — each executed tool call with its output.
- `mcp_approval_request` items — when `require_approval` demands one.

To approve a paused call, send a follow-up `responses.create` with an
`mcp_approval_response` input item and `previous_response_id`.

### `require_approval`

Supports the full OpenAI shape:

- `"never"` — execute all tools immediately.
- `"always"` — emit an `mcp_approval_request` for every call.
- `{ always: { tool_names: [...] }, never: { tool_names: [...] } }` — per-tool.

### Not yet supported

- OpenAI **connectors** (`connector_id`) — only raw `server_url` MCP servers.
- Built-in tools `web_search`, `file_search`, `code_interpreter`,
  `computer_use`, `image_generation`. Requests including them will fail at
  the backend since we forward them as-is.

## Configuration reference

All configuration is via constructor options — the library reads no
environment variables.

`new ResponsesClient(options)`:

| Option | Default | Description |
| --- | --- | --- |
| `backend` | — (required) | A `BackendAdapter` instance |
| `store` | `undefined` | A `Store` instance — required for conversations and `responses.{get,del}` |
| `maxIterations` | `10` | Hard cap on backend round-trips per request |

## Advanced usage

You can bypass the client and drive `AgentLoop` directly:

```ts
import { AgentLoop, OpenAICompatAdapter } from "responses-to-completions";

const agent = new AgentLoop({
  backend: new OpenAICompatAdapter({ baseUrl: "..." }),
});
const result = await agent.run({
  request: { model: "gpt-4o-mini", input: "hi" },
  history: [],
});
```

The translators (`itemsToMessages`, `completionToOutputItems`,
`translateChunkStream`) and the `resolveHistory` helper are also exported
for custom orchestration.

## Running the SDK smoke test

```bash
tsx examples/sdk-test.ts \
  --backend openai-compat \
  --base-url https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model gpt-4o-mini \
  --store-local ./.test-data
```

Without `--store-local`, conversation/persistence tests are skipped.

## Design notes

- **Items are the source of truth.** Conversations are persisted as an
  ordered array of typed items. Each request rehydrates items →
  chat-completions messages, runs the loop, appends new items.
- **Agent loop owns MCP execution.** Non-MCP function tools are passed
  through to the client (`function_call` items), same as OpenAI.
- **Streaming is synthesized.** Each `chat.completion.chunk` is mapped
  into the correct Responses-API event sequence; multi-turn tool loops
  serialize per iteration (stream deltas → execute tools → stream next).
- **No hidden writes.** Setting `store: false` on a request skips
  persistence; constructing without a `store` skips it for every request.

## License

MIT
