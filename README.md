# responses-to-completions

An **SDK** that exposes the OpenAI **Responses API** on top of any
OpenAI-compatible `/v1/chat/completions` backend (vLLM, Ollama, llama.cpp,
TGI, LiteLLM, Together, Groq, OpenRouter, …).

Construct a `ResponsesClient` with a `source` and its credentials, then call
`client.responses.create(...)` — the client builds the backend and store, and
handles conversation state, MCP tool execution, and streaming-event
translation internally.

There are three sources — `openAI`, `openRouter` and `ollama` — and each one
carries a default endpoint so its `config` needs only credentials. Any other
OpenAI-compatible server is reached through `source: "openAI"` by pointing
`baseUrl` at it; see
[Other OpenAI-compatible servers](#other-openai-compatible-servers) for the
one setting you have to change alongside it. `source` can also be left off
entirely for a [store-only client](#store-only-clients) that reads persisted
state without any provider credentials.

## Install

```bash
npm i responses-to-completions
```

## Migrating from 0.3

0.4.0 replaces the constructor's `backend`/`store` instances with
`source`/`config` plus `store`/`store_client`/`store_config` — the client now
builds adapters and stores itself. A 0.3.x construction and its 0.4.0
equivalent:

```ts
// 0.3.x
new ResponsesClient({
  backend: new OpenAICompatAdapter({
    baseUrl: "https://api.openai.com/v1",
    apiKey: process.env.OPENAI_API_KEY,
  }),
  store: new LocalFileStore("./.data"),
});

// 0.4.0
new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  store: true,
  store_client: "local",
  store_config: { dir: "./.data" },
});
```

| 0.3.x | 0.4.0 |
| --- | --- |
| `backend: new OpenAICompatAdapter({ apiKey })` | `source: "openAI", config: { apiKey }` |
| `backend: new OpenAICompatAdapter({ baseUrl })` — vLLM, llama.cpp, … | `source: "openAI", config: { baseUrl, maxTokensParam: "max_tokens" }` |
| `backend: new OpenRouterAdapter({ apiKey })` | `source: "openRouter", config: { apiKey }` |
| `backend: new OllamaAdapter({ host })` | `source: "ollama", config: { host, api: "native" }` |
| `store: new LocalFileStore(dir)` | `store: true, store_client: "local", store_config: { dir }` |
| `store: new S3Store({ bucket })` | `store: true, store_client: "S3", store_config: { bucket }` |
| no store | omit `store` (or `store: false`) |
| custom `BackendAdapter` | subclass `createBackend()`, or drive `AgentLoop` directly |

Behavior changes to know about: `source: "openRouter"` — and `"openAI"`
without a custom `baseUrl` — now requires `config.apiKey` at construction;
`source: "ollama"` defaults to Ollama's OpenAI-compatible `/v1` route
(`api: "native"` restores the old NDJSON adapter); and the new
`store_client: "openAI"` persists conversations in OpenAI's Conversations API
(available only for `source: "openAI"` with no custom `baseUrl`).

## Quick start

```ts
import { ResponsesClient } from "responses-to-completions";

const client = new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  // optional — required for conversations and responses.{get,del}
  store: true,
  store_client: "openAI",
});

const resp = await client.responses.create({
  model: "gpt-4o-mini",
  input: "Write a haiku about TypeScript.",
});
console.log(resp.output_text);
```

A client serves one `source`. The adapter and store are built for you from
`config` — no adapter or store classes to import. To reach two providers,
build two clients.

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

`finalResponse()` resolves after the turn is persisted and is where store
failures surface. The final stream event arrives *before* the store write
lands, so start follow-up turns on the same conversation (or read the turn
back) after `finalResponse()`, not inside the event loop.

### File and image inputs

`input_file` and `input_image` parts ride in a user message's content array:

```ts
const resp = await client.responses.create({
  model: "gpt-4o",
  input: [
    {
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: "Summarize this document." },
        {
          type: "input_file",
          filename: "contract.pdf",
          file_data: "data:application/pdf;base64,JVBERi0x...",
        },
      ],
    },
  ],
  stream: false,
});
```

How a document may be carried depends on the backend serving the turn:

| Carrier | `openAI` on chat-completions (default) | `openRouter` | `endpoint: "responses"` |
| --- | --- | --- | --- |
| `file_data` — base64 data URI | ✓ | ✓ | ✓ |
| `file_id` — an uploaded file | ✓ | — | ✓ |
| `file_url` — a plain URL | ✗ | ✓ (file-parser) | ✓ |

A `file_url` headed for a chat-completions server that only takes base64 is
rejected up front with an error naming the working alternatives, rather than
surfacing as an opaque upstream 400. Images use `input_image` with an
`image_url` (https or data URI) and an optional `detail`.

### Service tier

`service_tier` on a request is forwarded to the backend, and the response
reports the tier that actually served the turn — providers may downgrade
(e.g. `priority` past OpenAI's ramp rate limit). When the backend doesn't
report one, the requested tier is kept on the response.

## Backends

You never construct an adapter — `source` picks it and `config` configures it:

| `source` | Adapter | Default endpoint | Token-limit key |
| --- | --- | --- | --- |
| `openAI` | `OpenAICompatAdapter` | `https://api.openai.com/v1` | `max_completion_tokens` |
| `openRouter` | `OpenRouterAdapter` | `https://openrouter.ai/api/v1` | `max_tokens` |
| `ollama` | `OpenAICompatAdapter` | `http://localhost:11434/v1` | `max_tokens` |
| `ollama` + `api: "native"` | `OllamaAdapter` | `{host}/api/chat` | — |

The classes stay exported for driving `AgentLoop` directly, and
`createBackend()` can be overridden in a subclass to return any
`BackendAdapter` — see [Custom backends](#custom-backends).

### Other OpenAI-compatible servers

vLLM, llama.cpp's server, TGI, LiteLLM, Together, Groq and friends have no
`source` of their own. Reach them with `source: "openAI"` and a `baseUrl` —
but set `maxTokensParam` too:

```ts
const client = new ResponsesClient({
  source: "openAI",
  config: {
    baseUrl: "http://localhost:8000/v1",
    apiKey: "optional-bearer",
    maxTokensParam: "max_tokens", // ← required for non-OpenAI servers
  },
});
```

`source: "openAI"` sends `max_completion_tokens`, which is OpenAI's spelling
and the reason the source defaults to it. Most other implementations of the
API only understand `max_tokens` and will either reject the request or
silently ignore the cap, so `maxTokensParam: "max_tokens"` is not optional
when `baseUrl` points somewhere other than OpenAI. Everything else — MCP
tools, conversations, streaming, embeddings — behaves the same.

`config` also takes `headers`, `fetch` and `forceModel`, so a gateway that
wants its own auth header or a server that serves one fixed model needs no
custom adapter either.

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

Implement the `BackendAdapter` interface (`complete` + `stream`) and return it
from `createBackend()` in a subclass:

```ts
class MyClient extends ResponsesClient {
  protected override createBackend(): BackendAdapter {
    return new MyAdapter({ ... });
  }
}
const client = new MyClient({ source: "ollama", config: {} });
```

`createBackend()` and `createStore()` are called from the base constructor, so
a subclass's own field initializers have not run yet when they execute —
reading `this.something` inside them yields `undefined`. Build from the
`source`/`config` arguments they receive, or from module-level values:

```ts
import type { ClientSource, ProviderConfig } from "responses-to-completions";

class MyClient extends ResponsesClient {
  private opts = { timeout: 5000 };      // ✗ undefined during construction
  protected override createBackend(
    source: ClientSource,
    config: ProviderConfig,
  ): BackendAdapter {
    return new MyAdapter({ ...config }); // ✓ arguments are available
  }
}
```

## Configuration

```ts
const client = new ResponsesClient({
  source: "openAI",                              // "openAI" | "openRouter" | "ollama"
  config: { apiKey: process.env.OPENAI_API_KEY }, // typed by `source`
  store: true,
  store_client: "openAI",                        // "S3" | "local" | "openAI"
});
```

The types follow the values:

- **`config` is typed by `source`, and required with it.** Naming a `source`
  requires a `config` — `{}` for `ollama`, which needs no credentials — and a
  `config` without a `source` is rejected, since there would be nothing to
  configure. OpenRouter's routing keys (`provider`, `usageAccounting`,
  `appTitle`, `siteUrl`) are accepted on `openRouter` and rejected on the
  others; `ollama` gets `host` and `api`. Most sources take `apiKey`,
  `baseUrl`, `headers`, `fetch`, `forceModel`, `endpoint` and
  `maxTokensParam`, with two exceptions the types enforce: `openRouter` has
  no `maxTokensParam` (its adapter always sends `max_tokens`), and ollama's
  `api: "native"` mode drops `apiKey`/`baseUrl`/`endpoint`/`maxTokensParam`.
  Config keys are camelCase — one spelling per key — while the top-level
  store options (`store_client`, `store_config`) are snake_case.
  `openRouter` always requires a key; `openAI` requires one unless a custom
  `baseUrl` points it at a keyless OpenAI-compatible server.
- **`source` may be omitted for a store-only client.** See
  [Store-only clients](#store-only-clients).
- **`baseUrl` and `maxTokensParam` travel together.** Each source defaults
  both, so pointing `baseUrl` at a different server can leave the wrong
  token-limit key behind — see
  [Other OpenAI-compatible servers](#other-openai-compatible-servers).
- **`store_client` is typed by `store` and by `source`.** It is only accepted
  with `store: true`. `"S3"` and `"local"` work for every source; `"openAI"`
  only when `source` is `"openAI"` **without a custom `baseUrl`** — the OpenAI
  Conversations store ships every read and write to api.openai.com, so other
  providers' traffic must not use it, and a custom `baseUrl` names a compat
  server that has no Conversations API (both rejected at the type level and
  at construction). Without `store`, `conversations.*` and
  `responses.{get,del}` throw and `responses.create` runs stateless.
- **`store_config` is typed by `store_client`.** S3 requires a `bucket` (plus
  optional `prefix`, `client`, `clientConfig`); `"local"` requires a `dir`;
  the OpenAI store takes an optional
  `apiKey`/`baseUrl`/`headers`/`fetch`/`pageSize` and otherwise reuses the
  credentials already in `config` — except `baseUrl`, which is never
  inherited: conversations always go to api.openai.com unless
  `store_config.baseUrl` overrides it.

```ts
const client = new ResponsesClient({
  source: "openRouter",
  config: {
    apiKey: process.env.OPENROUTER_API_KEY,
    provider: { order: ["Anthropic", "Google"] },
  },
  store: true,
  store_client: "S3",
  store_config: { bucket: "my-bucket", prefix: "responses" },
});
```

Ollama is addressed by `host` — the server root, with `/v1` appended for you —
and `api: "native"` swaps the OpenAI-compatible route for the NDJSON
`/api/chat` one:

```ts
// OpenAI-compatible route (default). Same as baseUrl: "http://localhost:11434/v1".
new ResponsesClient({ source: "ollama", config: { host: "http://localhost:11434" } });

// Native /api/chat — NDJSON streaming, object-shaped tool arguments.
new ResponsesClient({ source: "ollama", config: { api: "native", host: "http://localhost:11434" } });
```

`api: "native"` is a different wire format, so the OpenAI-compatible keys
(`baseUrl`, `apiKey`, `endpoint`, `maxTokensParam`) are typed away on it —
it takes `host`, `headers`, `fetch` and `forceModel`. `baseUrl` wins over
`host` when both are given.

`config` is derived from the options of the adapter that serves the source
(`OpenAICompatAdapterOptions`, `OpenRouterAdapterOptions`,
`OllamaAdapterOptions`), so the two can't drift: a key the adapter doesn't
have is a compile error rather than a value silently dropped.

Both factories are `protected` methods — `createBackend(source, config)` and
`createStore(store_client, store_config, config)` — so a subclass can change
how a source is reached or plug in a store the built-ins don't cover. Both
are also exported standalone as `createBackendForSource()` and
`createStoreForClient()`. The S3 store is constructed lazily, so a client that
never persists to S3 never loads the AWS SDK.

## Embeddings

Call `client.embeddings.create(...)` to produce embeddings. The client
delegates to the configured backend adapter and returns the OpenAI-shaped
`EmbeddingsResponse` regardless of provider.

```ts
import { ResponsesClient } from "responses-to-completions";

const client = new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
});

const resp = await client.embeddings.create({
  model: "text-embedding-3-large",
  input: "hi",
});
// { object: "list", data: [{ object: "embedding", index, embedding }], model, usage }
```

Works with any source — change `source` and the call site stays the same:

```ts
import { ResponsesClient } from "responses-to-completions";

// Ollama — POST /api/embed, translated to the OpenAI shape
const ollamaClient = new ResponsesClient({
  source: "ollama",
  config: { api: "native", host: "http://localhost:11434" },
});
await ollamaClient.embeddings.create({
  model: "nomic-embed-text",
  input: ["a", "b", "c"],
});

// OpenRouter — routes to whichever provider exposes the model's embeddings
const orClient = new ResponsesClient({
  source: "openRouter",
  config: { apiKey: process.env.OPENROUTER_API_KEY! },
});
await orClient.embeddings.create({
  model: "openai/text-embedding-3-large",
  input: ["a", "b", "c"],
});
```

Embeddings don't require a `store`. If the source's adapter doesn't implement
`embeddings()`, the call throws.

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

Persistence is optional and off by default (`store: false`). Without it the
client still runs requests, but `conversations.*` and `responses.{get,del}`
throw and `responses.create` skips writing. It can also be the *only* half you
configure — see [Store-only clients](#store-only-clients).

Two different fields are spelled `store`, and they answer different questions.
The constructor's `store` decides *whether the client has a store at all*; the
request's `store` decides *whether one turn is written to it*. A turn that
should read a conversation but leave no trace is the second one — it still
needs a store to read through:

```ts
const client = new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  store: true,
  store_client: "openAI",
});

await client.responses.create({
  model: "gpt-5.6-luna",
  conversation: "conv_…",   // history is loaded
  input: "…",
  store: false,             // this turn is not persisted
});
```

So `conversation` and `previous_response_id` require the *constructor* option.
A client built with `store: false` has nothing to resolve them against, and
its type says so — the flag is inferred from the `store` literal, so the two
fields are rejected at compile time rather than at the call:

```ts
const storeless = new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  store: false,
});

await storeless.responses.create({
  model: "gpt-5.6-luna",
  input: "…",
  conversation: "conv_…",  // ← Type 'string' is not assignable to type 'undefined'
});
```

The inference reads the literal you passed, so it applies to `store: false`
and `store: true`. Omitting `store` altogether (and any client whose
construction isn't in view — a bare `ResponsesClient` annotation, a subclass)
keeps the permissive `boolean` flag and the runtime check. `CreateRequestFor`
is exported if you want to name either request shape.

### `store_client: "S3"` — for production

```ts
new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  store: true,
  store_client: "S3",
  store_config: {
    bucket: "my-bucket",
    prefix: "prod/responses",
    clientConfig: { region: "us-east-1" },
  },
});
```

One JSON object per artifact, under
`<prefix>/conversations/<id>.json` and `<prefix>/responses/<id>.json`. The
`S3Store` is constructed on first use, so `@aws-sdk/client-s3` is never loaded
by a client that doesn't store to S3.

### `store_client: "openAI"` — provider-owned state

Only for `source: "openAI"`. Conversation state lives in OpenAI's
Conversations API and `conversations.create` returns the real `conv_…` id
OpenAI issued. `store_config` is optional — the credentials in `config` are
reused:

```ts
new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY },
  store: true,
  store_client: "openAI",
});
```

Every read and write sends conversation content to api.openai.com.

Responses are the one asymmetry: OpenAI persists the ones its own `/responses`
endpoint produced, so this store's `saveResponse` is a no-op and `getResponse`
reads back through to the provider. What `previous_response_id` can resolve
therefore follows the endpoint the client is pointed at:

```ts
new ResponsesClient({
  source: "openAI",
  config: { apiKey: process.env.OPENAI_API_KEY, endpoint: "responses" },
  store: true,
  store_client: "openAI",
});
```

- `endpoint: "responses"` — OpenAI serves the turn and issues its id, which
  the client adopts as `response.id` instead of a locally generated one.
  `previous_response_id`, `responses.get` and `responses.del` all resolve
  against OpenAI afterwards.
- `endpoint: "completions"` (the default) — the response is synthesized here
  from a chat-completions turn and OpenAI has no copy, so
  `previous_response_id` cannot resolve. Continue with `conversation`, which
  this store persists either way.

A response OpenAI served carries no record of the SDK-side conversation it
belonged to, so chaining it by `previous_response_id` replays that turn's
output rather than the whole conversation. Pass `conversation` when the full
transcript matters. The `"local"` and `"S3"` stores persist responses
themselves and have neither restriction.

### `store_client: "local"` — for development and tests

```ts
new ResponsesClient({
  source: "ollama",
  config: { host: "http://localhost:11434" },
  store: true,
  store_client: "local",
  store_config: { dir: "./.data" },
});
```

One JSON file per artifact, under `<dir>/conversations/<id>.json` and
`<dir>/responses/<id>.json`.

### Store-only clients

Omit `source` to build a client that reads and writes persisted state without
reaching a provider at all — no credentials, no adapter:

```ts
const reader = new ResponsesClient({
  store: true,
  store_client: "S3",
  store_config: { bucket: "my-bucket", prefix: "prod/responses" },
});

const { data } = await reader.conversations.items.list("user-42-thread");
const resp = await reader.responses.get("resp_abc");
```

`conversations.*` and `responses.{get,del}` work as usual.
`responses.create` and `embeddings.create` throw, having nothing to call.
Useful for an admin tool, a transcript viewer, a retention job, or a test that
exercises persistence without a model server.

The type enforces both halves of the rule: naming a `source` requires its
`config`, and omitting `source` requires a real store — `store: true` with
`"S3"` or `"local"`. A client with neither could do nothing at all, so it
won't compile, and `store_client: "openAI"` is unavailable here because it has
no source to borrow credentials from.

```ts
new ResponsesClient({});                        // ✗ neither half
new ResponsesClient({ source: "ollama" });      // ✗ source without config
new ResponsesClient({ config: { apiKey: k } }); // ✗ config without source
new ResponsesClient({ store: true, store_client: "openAI" }); // ✗ no credentials
```

### Custom stores

`LocalFileStore`, `S3Store` and `OpenAIConversationStore` are exported for
direct use, and any `Store` implementation (see `src/store/store.ts`) can be
returned from `createStore()` in a subclass.

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
| `conversations.create({ id?, items?, metadata? })` | `id` is optional; client-generated when omitted, and rejected on `store_client: "openAI"` |
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

Who assigns the id differs by store. `"local"` and `"S3"` own their keyspace,
so any id you pass is valid and a `conversation` the store hasn't seen is
created on first use — you can mint ids yourself and hand them to
`responses.create` directly. On `"openAI"` the provider assigns ids:
`conversations.create()` rejects a supplied `id`, and passing a `conversation`
OpenAI doesn't recognize throws rather than creating it. Take the id from
`conversations.create()` and pass that:

```ts
// "local" / "S3" — your id, created on first use.
await client.responses.create({ model, conversation: "user-42-thread", input });

// "openAI" — the provider's id.
const conv = await client.conversations.create({ metadata: { user: "u_42" } });
await client.responses.create({ model, conversation: conv.id, input });
```

Everything after that — `items.list`, `items.append`, `update`, `del`, and
transcript replay on each turn — behaves identically across all three.

You can also continue by `previous_response_id`, same as OpenAI. That needs a
store the response can be read back from: `"local"` and `"S3"` persist
responses themselves, while `"openAI"` reads through to the provider and so
only resolves ids OpenAI issued — see
[`store_client: "openAI"`](#store_client-openai--provider-owned-state).

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
| `source` | — (omit for [store-only](#store-only-clients)) | `"openAI"`, `"openRouter"` or `"ollama"`. Other OpenAI-compatible servers go through `"openAI"` with a `baseUrl` — [see above](#other-openai-compatible-servers) |
| `config` | — (required with `source`, rejected without) | Credentials and options for that source's adapter; `{}` for `ollama` |
| `store` | `false` | `true` to persist — required for conversations and `responses.{get,del}`, and required outright when `source` is omitted |
| `store_client` | — (required with `store`) | `"S3"`, `"local"`, or `"openAI"` when `source` is `"openAI"` |
| `store_config` | — (required for `"S3"`/`"local"`) | `{ bucket, prefix?, … }`, `{ dir }`, or the OpenAI store's options |
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
