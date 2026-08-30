import type { ResponsesClientOptions } from "../src/index.js";

/** The `--backend` flag the example CLIs accept. */
export type BackendKind = "openai-compat" | "ollama" | "openrouter";

export interface ClientArgs {
  backend: BackendKind;
  baseUrl?: string;
  apiKey?: string;
  /** Directory for the local file store; omit to run without persistence. */
  storeLocal?: string;
  /** Call the upstream `/responses` endpoint instead of `/chat/completions`. */
  responsesEndpoint?: boolean;
  /** Drive Ollama's native NDJSON `/api/chat` route instead of its `/v1`. */
  ollamaNative?: boolean;
}

/**
 * CLI flags to constructor options. `--backend` names the source, the rest of
 * the flags fill in that source's config, and `--store-local` turns on the
 * local file store.
 */
export function clientOptions(args: ClientArgs): ResponsesClientOptions {
  const endpoint = args.responsesEndpoint
    ? ({ endpoint: "responses" } as const)
    : {};
  const store = args.storeLocal
    ? ({
        store: true,
        store_client: "local",
        store_config: { dir: args.storeLocal },
      } as const)
    : ({ store: false } as const);

  if (args.backend === "ollama") {
    // Ollama serves /v1/chat/completions but no /v1/responses, so forwarding
    // the flag would 404 every request (see src/backend/from-source.ts).
    if (args.responsesEndpoint) {
      throw new Error(
        "--responses-endpoint is not supported for ollama — it has no /v1/responses route",
      );
    }
    // `--ollama-native` maps to `api: "native"`, the NDJSON /api/chat route.
    if (args.ollamaNative) {
      return {
        source: "ollama",
        config: {
          api: "native",
          ...(args.baseUrl ? { host: args.baseUrl } : {}),
        },
        ...store,
      };
    }
    return {
      source: "ollama",
      config: {
        ...(args.baseUrl ? { host: args.baseUrl } : {}),
        ...(args.apiKey ? { apiKey: args.apiKey } : {}),
      },
      ...store,
    };
  }

  if (args.backend === "openrouter") {
    if (!args.apiKey) throw new Error("--api-key is required for openrouter");
    return {
      source: "openRouter",
      config: {
        apiKey: args.apiKey,
        ...(args.baseUrl ? { baseUrl: args.baseUrl } : {}),
        ...endpoint,
      },
      ...store,
    };
  }

  // The flag names an OpenAI-compatible server, so the target must be
  // explicit: silently defaulting to api.openai.com would send the prompt
  // (and key) to the wrong backend. Reaching OpenAI itself is
  // `--base-url https://api.openai.com/v1`. The key is optional — keyless
  // local servers (vLLM, llama.cpp, …) take none, and OpenAI rejects keyless
  // requests on its own.
  if (!args.baseUrl) {
    throw new Error("--base-url is required for openai-compat");
  }
  return {
    source: "openAI",
    config: {
      ...(args.apiKey ? { apiKey: args.apiKey } : {}),
      baseUrl: args.baseUrl,
      ...endpoint,
    },
    ...store,
  };
}
