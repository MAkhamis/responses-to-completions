/**
 * End-to-end SDK smoke test.
 *
 * Exercises ResponsesClient against a real backend. The store flag is
 * optional; tests that require persistence are skipped when omitted.
 *
 *   tsx examples/sdk-test.ts \
 *     --backend openai-compat \
 *     --base-url https://api.openai.com/v1 \
 *     --api-key sk-... \
 *     --model gpt-4o-mini \
 *     [--store-local ./.test-data]
 */
import {
  LocalFileStore,
  OllamaAdapter,
  OpenAICompatAdapter,
  OpenRouterAdapter,
  ResponsesClient,
  type BackendAdapter,
  type Store,
} from "../src/index.js";

export interface Args {
  backend: "openai-compat" | "ollama" | "openrouter";
  baseUrl?: string;
  apiKey?: string;
  model: string;
  storeLocal?: string;
}

interface TestResult {
  name: string;
  status: "pass" | "fail" | "skip";
  detail?: string;
}

export async function main(args: Args): Promise<number> {
  const backend = buildBackend(args);
  const store: Store | undefined = args.storeLocal
    ? new LocalFileStore(args.storeLocal)
    : undefined;

  const client = new ResponsesClient({ backend, store });
  const model = args.model;
  const results: TestResult[] = [];

  // --- Test A: non-streaming -------------------------------------------
  results.push(
    await runTest("A: non-streaming create", async () => {
      const resp = await client.responses.create({
        model,
        input: "Say hi in five words or fewer.",
      });
      assert(resp.status === "completed", `status was ${resp.status}`);
      assert(
        typeof resp.output_text === "string" && resp.output_text.length > 0,
        "output_text was empty",
      );
      return `id=${resp.id} text=${truncate(resp.output_text ?? "", 60)}`;
    }),
  );

  // --- Test B: streaming -----------------------------------------------
  let streamedFinalId: string | null = null;
  results.push(
    await runTest("B: streaming create", async () => {
      const stream = await client.responses.create({
        model,
        input: "Stream the word 'hello' three times, separated by commas.",
        stream: true,
      });
      let accumulated = "";
      let deltaCount = 0;
      for await (const ev of stream) {
        if (ev.type === "response.output_text.delta") {
          accumulated += ev.delta;
          deltaCount++;
        }
      }
      const finalResp = await stream.finalResponse();
      streamedFinalId = finalResp.id;
      assert(deltaCount > 0, "no output_text.delta events received");
      assert(
        finalResp.status === "completed",
        `status was ${finalResp.status}`,
      );
      assert(
        (finalResp.output_text ?? "").length > 0,
        "final output_text was empty",
      );
      const matches = (finalResp.output_text ?? "").includes(
        accumulated.slice(0, 20),
      );
      assert(matches, "accumulated deltas did not match final output_text");
      return `id=${finalResp.id} deltas=${deltaCount} text=${truncate(finalResp.output_text ?? "", 60)}`;
    }),
  );

  // --- Test C: conversation continuation (needs store) -----------------
  let convId: string | null = null;
  if (!store) {
    results.push({
      name: "C: conversation continuation",
      status: "skip",
      detail: "no --store-local provided",
    });
  } else {
    results.push(
      await runTest("C: conversation continuation", async () => {
        const conv = await client.conversations.create({
          metadata: { test: "sdk-smoke" },
        });
        convId = conv.id;
        await client.responses.create({
          model,
          conversation: conv.id,
          input:
            "Remember the secret word 'banana'. Acknowledge in one short sentence.",
        });
        const second = await client.responses.create({
          model,
          conversation: conv.id,
          input: "What was the secret word? Reply with just the word.",
        });
        assert(second.status === "completed", `status=${second.status}`);
        const text = (second.output_text ?? "").toLowerCase();
        assert(
          text.includes("banana"),
          `second turn did not recall context (got: ${truncate(second.output_text ?? "", 80)})`,
        );
        return `conv=${conv.id} text=${truncate(second.output_text ?? "", 60)}`;
      }),
    );
  }

  // --- Test D: items.list (needs store + conversation) -----------------
  if (!store || !convId) {
    results.push({
      name: "D: conversations.items.list",
      status: "skip",
      detail: !store ? "no --store-local provided" : "conversation test failed",
    });
  } else {
    results.push(
      await runTest("D: conversations.items.list", async () => {
        const page = await client.conversations.items.list(convId!);
        assert(page.data.length > 0, "items.list returned 0 entries");
        return `count=${page.data.length}`;
      }),
    );
  }

  // --- Test E: responses.get + responses.del ---------------------------
  if (!store) {
    results.push({
      name: "E: responses.get + del",
      status: "skip",
      detail: "no --store-local provided",
    });
  } else if (!streamedFinalId) {
    results.push({
      name: "E: responses.get + del",
      status: "skip",
      detail: "streaming test did not produce a response id",
    });
  } else {
    results.push(
      await runTest("E: responses.get + del", async () => {
        const fetched = await client.responses.get(streamedFinalId!);
        assert(fetched !== null, "responses.get returned null");
        assert(
          fetched!.id === streamedFinalId,
          `id mismatch: ${fetched!.id} vs ${streamedFinalId}`,
        );
        const deleted = await client.responses.del(streamedFinalId!);
        assert(deleted.deleted, "responses.del reported deleted=false");
        const reFetched = await client.responses.get(streamedFinalId!);
        assert(reFetched === null, "response still present after delete");
        return `id=${streamedFinalId} deleted=true`;
      }),
    );
  }

  // --- Summary ---------------------------------------------------------
  printSummary(results);
  const failed = results.filter((r) => r.status === "fail");
  return failed.length === 0 ? 0 : 1;
}

// ---- helpers -------------------------------------------------------------

export function parseArgs(argv: string[]): Args {
  const get = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    if (i === -1) return undefined;
    const v = argv[i + 1];
    if (v === undefined || v.startsWith("--")) {
      throw new Error(`--${name} requires a value`);
    }
    return v;
  };
  const backend = (get("backend") ?? "openai-compat") as Args["backend"];
  if (!["openai-compat", "ollama", "openrouter"].includes(backend)) {
    throw new Error(`unknown --backend: ${backend}`);
  }
  const model = get("model");
  if (!model) throw new Error("--model is required");
  return {
    backend,
    baseUrl: get("base-url"),
    apiKey: get("api-key"),
    model,
    storeLocal: get("store-local"),
  };
}

function buildBackend(args: Args): BackendAdapter {
  if (args.backend === "ollama") {
    return new OllamaAdapter({ host: args.baseUrl });
  }
  if (args.backend === "openrouter") {
    if (!args.apiKey) throw new Error("--api-key is required for openrouter");
    return new OpenRouterAdapter({ apiKey: args.apiKey });
  }
  if (!args.baseUrl)
    throw new Error("--base-url is required for openai-compat");
  return new OpenAICompatAdapter({
    baseUrl: args.baseUrl,
    apiKey: args.apiKey,
  });
}

async function runTest(
  name: string,
  fn: () => Promise<string | void>,
): Promise<TestResult> {
  const t0 = Date.now();
  try {
    const detail = (await fn()) ?? undefined;
    const ms = Date.now() - t0;
    console.log(`✓ ${name} (${ms}ms)${detail ? ` — ${detail}` : ""}`);
    return { name, status: "pass", detail };
  } catch (err) {
    const ms = Date.now() - t0;
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`✗ ${name} (${ms}ms) — ${msg}`);
    return { name, status: "fail", detail: msg };
  }
}

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

function printSummary(results: TestResult[]): void {
  const pass = results.filter((r) => r.status === "pass").length;
  const fail = results.filter((r) => r.status === "fail").length;
  const skip = results.filter((r) => r.status === "skip").length;
  console.log("");
  console.log("─".repeat(60));
  for (const r of results) {
    const marker =
      r.status === "pass" ? "PASS" : r.status === "fail" ? "FAIL" : "SKIP";
    console.log(
      `  ${marker.padEnd(5)} ${r.name}${r.detail && r.status !== "pass" ? `  (${r.detail})` : ""}`,
    );
  }
  console.log("─".repeat(60));
  console.log(`  ${pass} passed, ${fail} failed, ${skip} skipped`);
}

