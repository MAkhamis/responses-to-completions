/**
 * Interactive streaming chat REPL — a realistic end-to-end SDK use case.
 *
 * Builds a ResponsesClient against a chat-completions backend, creates a
 * persistent conversation in a local file store, and runs a multi-turn
 * stdin/stdout chat loop. Each turn streams tokens as they arrive and
 * appends the exchange to the conversation so the next turn has context.
 *
 *   tsx examples/sdk-script.ts
 *
 * Commands:
 *   /stream   toggle between streaming and non-streaming responses
 *   /history  dump conversation items as JSON (if a conversation is active)
 *   /exit     quit (or Ctrl-D)
 */
import * as readline from "node:readline/promises";
import { stdin, stdout, exit } from "node:process";
import { ResponsesClient } from "../src/index.js";
import { clientOptions } from "./build-client.js";

const BASE_URL = "BASE_URL";
const API_KEY = "API-KEY";
const MODEL = "gpt-4o-mini";

async function chat(
  backendKind: "openrouter" | "openai" | "ollam",
  streaming: boolean,
  store: boolean,
): Promise<void> {
  const client = new ResponsesClient(
    clientOptions({
      backend:
        backendKind === "openrouter"
          ? "openrouter"
          : backendKind === "openai"
            ? "openai-compat"
            : "ollama",
      baseUrl: BASE_URL,
      apiKey: API_KEY,
      storeLocal: ".data",
    }),
  );

  const conv = store ? await client.conversations.create({}) : null;

  console.log(`conversation: ${conv?.id}`);
  console.log(`model:        ${MODEL}`);
  console.log(`(type /exit to quit, /history to dump items)\n`);

  const rl = readline.createInterface({ input: stdin, output: stdout });

  while (true) {
    const prompt = `you (${streaming ? "stream" : "non-stream"}) ▸ `;
    const user = (await rl.question(prompt)).trim();
    if (!user) continue;
    if (user === "/exit") break;
    if (user === "/stream") {
      streaming = !streaming;
      console.log(`→ ${streaming ? "streaming" : "non-streaming"} mode\n`);
      continue;
    }
    if (user === "/history" && conv) {
      const page = await client.conversations.items.list(conv.id);
      console.log(JSON.stringify(page.data, null, 2));
      continue;
    }

    if (streaming) {
      stdout.write("bot ▸ ");
      const stream = await client.responses.create({
        model: MODEL,
        conversation: conv?.id,
        input: user,
        stream: true,
      });
      const counts = new Map<string, number>();
      for await (const ev of stream) {
        counts.set(ev.type, (counts.get(ev.type) ?? 0) + 1);
        if (ev.type === "response.output_text.delta") stdout.write(ev.delta);
      }
      const final = await stream.finalResponse();
      stdout.write(`\n${footer(final.id, final.usage)}\n`);
      stdout.write(`  └─ events: ${summarizeEvents(counts)}\n\n`);
    } else {
      const resp = await client.responses.create({
        model: MODEL,
        conversation: conv?.id,
        input: user,
      });
      stdout.write(`bot ▸ ${resp.output_text ?? ""}\n`);
      stdout.write(`${footer(resp.id, resp.usage)}\n\n`);
    }
  }

  rl.close();
  console.log(`\nconversation saved as ${conv?.id}`);
}

function footer(
  id: string,
  usage: { input_tokens: number; output_tokens: number } | null,
): string {
  const tokens = usage
    ? ` (${usage.input_tokens}→${usage.output_tokens} tokens)`
    : "";
  return `  └─ ${id}${tokens}`;
}

function summarizeEvents(counts: Map<string, number>): string {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([type, n]) => `${type.replace(/^response\./, "")}×${n}`)
    .join(", ");
}

chat("openrouter", true, true).catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  exit(1);
});
