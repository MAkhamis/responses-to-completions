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
    return {
      source: "ollama",
      config: {
        ...(args.baseUrl ? { host: args.baseUrl } : {}),
        ...(args.apiKey ? { apiKey: args.apiKey } : {}),
        ...endpoint,
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

  if (!args.apiKey) throw new Error("--api-key is required for openai-compat");
  return {
    source: "openAI",
    config: {
      apiKey: args.apiKey,
      ...(args.baseUrl ? { baseUrl: args.baseUrl } : {}),
      ...endpoint,
    },
    ...store,
  };
}
