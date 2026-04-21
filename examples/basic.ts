/**
 * Example: run the proxy against Ollama's OpenAI-compat endpoint with a
 * local file store, then hit it with the OpenAI SDK.
 *
 *   pnpm tsx examples/basic.ts
 *   # in another terminal:
 *   curl -s http://localhost:8787/v1/responses \
 *     -H 'content-type: application/json' \
 *     -d '{"model":"llama3.1","input":"hi"}' | jq
 */
import { createServer, OpenAICompatAdapter, LocalFileStore } from "../src/index.js";

const app = createServer({
  backend: new OpenAICompatAdapter({
    baseUrl: process.env.BACKEND_BASE_URL ?? "http://localhost:11434/v1",
    apiKey: process.env.BACKEND_API_KEY,
  }),
  store: new LocalFileStore(process.env.STORE_LOCAL_ROOT ?? "./.data"),
  maxIterations: 8,
});

const port = parseInt(process.env.PORT ?? "8787", 10);
app.listen(port, () => {
  console.log(`Responses-to-completions proxy on http://localhost:${port}`);
});
