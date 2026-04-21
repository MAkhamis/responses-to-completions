#!/usr/bin/env node
/**
 * CLI: `responses-to-completions`
 *
 * Reads config from env vars. Starts an HTTP server that speaks the Responses
 * API on the configured port, forwarding to a chat-completions backend and
 * persisting state via a pluggable store.
 *
 * Env vars:
 *   PORT                       (default 8787)
 *   HOST                       (default 0.0.0.0)
 *   BACKEND                    "openai-compat" | "ollama" (default openai-compat)
 *   BACKEND_BASE_URL           required for openai-compat (e.g. https://api.openai.com/v1)
 *   BACKEND_API_KEY            optional bearer token
 *   OLLAMA_HOST                default http://localhost:11434
 *   FORCE_MODEL                optional — override model on every upstream call
 *   STORE                      "local" | "s3" (default local)
 *   STORE_LOCAL_ROOT           default ./.data
 *   STORE_S3_BUCKET            required for s3
 *   STORE_S3_PREFIX            optional
 *   STORE_S3_REGION            optional (falls back to AWS SDK defaults)
 *   MAX_ITERATIONS             default 10
 */
import { createServer } from "./server/routes.js";
import { OpenAICompatAdapter, OllamaAdapter } from "./backend/index.js";
import { LocalFileStore, S3Store } from "./store/index.js";

function envRequired(name: string): string {
  const v = process.env[name];
  if (!v) {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return v;
}

function main(): void {
  const port = parseInt(process.env.PORT ?? "8787", 10);
  const host = process.env.HOST ?? "0.0.0.0";

  const backendKind = (process.env.BACKEND ?? "openai-compat").toLowerCase();
  const forceModel = process.env.FORCE_MODEL;

  let backend;
  if (backendKind === "ollama") {
    backend = new OllamaAdapter({
      host: process.env.OLLAMA_HOST,
      ...(forceModel ? { forceModel } : {}),
    });
  } else {
    backend = new OpenAICompatAdapter({
      baseUrl: envRequired("BACKEND_BASE_URL"),
      apiKey: process.env.BACKEND_API_KEY,
      ...(forceModel ? { forceModel } : {}),
    });
  }

  const storeKind = (process.env.STORE ?? "local").toLowerCase();
  let store;
  if (storeKind === "s3") {
    store = new S3Store({
      bucket: envRequired("STORE_S3_BUCKET"),
      prefix: process.env.STORE_S3_PREFIX,
      clientConfig: process.env.STORE_S3_REGION ? { region: process.env.STORE_S3_REGION } : {},
    });
  } else {
    store = new LocalFileStore(process.env.STORE_LOCAL_ROOT ?? "./.data");
  }

  const maxIterations = parseInt(process.env.MAX_ITERATIONS ?? "10", 10);
  const app = createServer({ backend, store, maxIterations });

  app.listen(port, host, () => {
    console.log(
      `[responses-to-completions] listening on http://${host}:${port}\n` +
        `  backend: ${backend.name}${forceModel ? ` (force model: ${forceModel})` : ""}\n` +
        `  store:   ${storeKind}`,
    );
  });
}

main();
