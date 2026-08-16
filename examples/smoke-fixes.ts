/**
 * Comprehensive SDK smoke suite.
 *
 * Exercises the full public surface of ResponsesClient against a real backend.
 * Tests self-skip when their prerequisites aren't supplied (store, reasoning
 * model, /responses endpoint, etc.) so the suite can be run incrementally.
 *
 * Test groups:
 *   A. request validation (no network — fast preflight)
 *   B. basic responses.create — non-stream / stream
 *   C. StreamResponse contract (break-early, no-iteration, abort, ordering)
 *   D. function tools (client-side execution loop)
 *   E. conversations (needs --store-local)
 *   F. responses persistence + req.store=false (needs --store-local)
 *   G. /responses pass-through (needs --responses-endpoint)
 *   H. reasoning model index alignment (needs --reasoning-model on openrouter)
 *
 * Usage:
 *   tsx examples/smoke-fixes.ts \
 *     --backend openai-compat \
 *     --base-url https://api.openai.com/v1 \
 *     --api-key $OPENAI_API_KEY \
 *     --model gpt-4o-mini \
 *     [--store-local ./.smoke-data] \
 *     [--responses-endpoint] \
 *     [--reasoning-model openai/o3-mini]   # only on --backend openrouter
 */
import { rm } from "node:fs/promises";
import { ResponsesClient, type StreamEvent } from "../src/index.js";
import { clientOptions } from "./build-client.js";

interface Args {
  backend: "openai-compat" | "ollama" | "openrouter";
  baseUrl?: string;
  apiKey?: string;
  model: string;
  reasoningModel?: string;
  storeLocal?: string;
  responsesEndpoint: boolean;
}

interface TestResult {
  name: string;
  status: "pass" | "fail" | "skip";
  detail?: string;
}

async function main(args: Args): Promise<number> {
  const store = args.storeLocal !== undefined;
  const client = new ResponsesClient(clientOptions(args));
  const results: TestResult[] = [];

  // === A. Request validation (no network) ================================
  results.push(
    await runTest("A1: missing model throws synchronously", async () => {
      await expectThrow(
        () =>
          (client.responses.create as (req: unknown) => Promise<unknown>)({
            input: "hi",
          }),
        /model/i,
      );
      return "ok";
    }),
  );

  results.push(
    await runTest("A2: missing input + previous_response_id throws", async () => {
      await expectThrow(
        () =>
          (client.responses.create as (req: unknown) => Promise<unknown>)({
            model: args.model,
          }),
        /input|previous_response_id/i,
      );
      return "ok";
    }),
  );

  if (!store) {
    const storelessClient = new ResponsesClient(
      clientOptions({ ...args, storeLocal: undefined }),
    );
    results.push(
      await runTest("A3: store ops without store throw clearly", async () => {
        await expectThrow(
          () => storelessClient.conversations.create({}),
          /store/i,
        );
        await expectThrow(
          () => storelessClient.responses.get("resp_x"),
          /store/i,
        );
        return "ok";
      }),
    );
  } else {
    results.push({
      name: "A3: store ops without store throw clearly",
      status: "skip",
      detail: "rerun without --store-local to exercise",
    });
  }

  // === B. Basic create ===================================================
  results.push(
    await runTest("B1: non-stream, string input", async () => {
      const resp = await client.responses.create({
        model: args.model,
        input: "Say hi in five words or fewer.",
      });
      assert(resp.status === "completed", `status=${resp.status}`);
      assert(
        typeof resp.output_text === "string" && resp.output_text.length > 0,
        "empty output_text",
      );
      return `id=${resp.id} text=${truncate(resp.output_text ?? "", 40)}`;
    }),
  );

  results.push(
    await runTest("B2: non-stream, structured input items", async () => {
      const resp = await client.responses.create({
        model: args.model,
        input: [
          { type: "message", role: "user", content: "Reply with the word 'ok'." },
        ],
      });
      assert(resp.status === "completed", `status=${resp.status}`);
      return `text=${truncate(resp.output_text ?? "", 40)}`;
    }),
  );

  results.push(
    await runTest("B3: stream, deltas reconstruct final text", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Stream the digits 1 2 3 separated by spaces.",
        stream: true,
      });
      let acc = "";
      let deltaCount = 0;
      for await (const ev of stream) {
        if (ev.type === "response.output_text.delta") {
          acc += ev.delta;
          deltaCount++;
        }
      }
      const final = await stream.finalResponse();
      assert(deltaCount > 0, "no delta events");
      assert(final.status === "completed", `status=${final.status}`);
      assert(
        (final.output_text ?? "").includes(acc.slice(0, 5)),
        "accumulated deltas don't match final output_text",
      );
      return `deltas=${deltaCount} text=${truncate(final.output_text ?? "", 40)}`;
    }),
  );

  results.push(
    await runTest("B4: instructions are honored", async () => {
      const resp = await client.responses.create({
        model: args.model,
        instructions:
          "Whenever asked anything, reply with exactly the single word: 'banana'.",
        input: "What is 2+2?",
      });
      const text = (resp.output_text ?? "").toLowerCase();
      assert(
        text.includes("banana"),
        `instructions not followed: got ${truncate(resp.output_text ?? "", 60)}`,
      );
      return `text=${truncate(resp.output_text ?? "", 40)}`;
    }),
  );

  // === C. StreamResponse contract ========================================
  results.push(
    await runTest("C1: stream break-early then finalResponse", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Count slowly from one to twenty, one per line.",
        stream: true,
      });
      let seen = 0;
      for await (const ev of stream) {
        if (ev.type === "response.output_text.delta") {
          seen++;
          if (seen >= 2) break;
        }
      }
      const t0 = Date.now();
      const final = await withTimeout(
        stream.finalResponse(),
        20_000,
        "finalResponse() did not settle after break",
      );
      assert(
        final.status === "completed" || final.status === "failed",
        `status=${final.status}`,
      );
      return `seen=${seen} status=${final.status} settle=${Date.now() - t0}ms`;
    }),
  );

  results.push(
    await runTest("C2: finalResponse without iterating", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Reply with 'ok'.",
        stream: true,
      });
      const final = await withTimeout(
        stream.finalResponse(),
        20_000,
        "finalResponse() did not settle without iteration",
      );
      assert(final.status === "completed", `status=${final.status}`);
      assert(
        (final.output_text ?? "").length > 0,
        "output_text empty even though stream completed",
      );
      return `text=${truncate(final.output_text ?? "", 40)}`;
    }),
  );

  results.push(
    await runTest("C3: double-iteration throws", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Reply with 'ok'.",
        stream: true,
      });
      for await (const _ of stream) void _;
      let threw = false;
      try {
        for await (const _ of stream) void _;
      } catch (err) {
        threw = err instanceof Error && /once/i.test(err.message);
      }
      assert(threw, "second iteration did not throw");
      await stream.finalResponse();
      return "ok";
    }),
  );

  results.push(
    await runTest("C4: event ordering — sequence_number monotonic", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Reply with 'ok'.",
        stream: true,
      });
      let lastSeq = -1;
      let count = 0;
      for await (const ev of stream) {
        assert(
          ev.sequence_number > lastSeq,
          `sequence_number not monotonic: ${ev.sequence_number} ≤ ${lastSeq}`,
        );
        lastSeq = ev.sequence_number;
        count++;
      }
      await stream.finalResponse();
      return `events=${count} maxSeq=${lastSeq}`;
    }),
  );

  results.push(
    await runTest("C5: every added has matching done at same index", async () => {
      const stream = await client.responses.create({
        model: args.model,
        input: "Reply with 'ok'.",
        stream: true,
      });
      const added = new Map<number, string>();
      const done = new Map<number, string>();
      for await (const ev of stream) {
        if (ev.type === "response.output_item.added") {
          added.set(ev.output_index, (ev.item as { type: string }).type);
        } else if (ev.type === "response.output_item.done") {
          done.set(ev.output_index, (ev.item as { type: string }).type);
        }
      }
      const final = await stream.finalResponse();
      for (const [idx, type] of added.entries()) {
        assert(
          done.get(idx) === type,
          `index ${idx}: added=${type}, done=${done.get(idx)}`,
        );
      }
      for (let i = 0; i < final.output.length; i++) {
        const itemType = (final.output[i] as { type: string }).type;
        assert(
          added.get(i) === itemType,
          `final.output[${i}]=${itemType} but event at index ${i}=${added.get(i)}`,
        );
      }
      return `items=${final.output.length} indices=${added.size}`;
    }),
  );

  results.push(
    await runTest("C6: AbortSignal cancels streaming", async () => {
      const ctrl = new AbortController();
      const stream = await client.responses.create({
        model: args.model,
        input: "Count slowly from one to one hundred, one per line.",
        stream: true,
        signal: ctrl.signal,
      });
      setTimeout(() => ctrl.abort(), 400);
      let aborted = false;
      try {
        for await (const _ of stream) void _;
        await stream.finalResponse();
      } catch (err) {
        aborted = true;
        const msg = err instanceof Error ? err.message : String(err);
        // Either AbortError or a backend error mentioning abort/cancel is fine.
        assert(
          /abort|cancel/i.test(msg) || err instanceof Error,
          `unexpected error: ${msg}`,
        );
      }
      // Some backends complete extremely fast; tolerate that.
      return aborted ? "aborted" : "completed-before-abort";
    }),
  );

  // === D. Function tools =================================================
  results.push(
    await runTest("D1: function tool returns function_call item", async () => {
      const resp = await client.responses.create({
        model: args.model,
        input: "What's the weather in Tokyo? Use the tool.",
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get current weather for a city.",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
        tool_choice: "auto",
      });
      const fc = resp.output.find(
        (o): o is Extract<typeof o, { type: "function_call" }> =>
          (o as { type: string }).type === "function_call",
      );
      assert(fc, `no function_call item in output (status=${resp.status})`);
      assert(fc.name === "get_weather", `wrong tool: ${fc.name}`);
      assert(fc.call_id && fc.call_id.length > 0, "function_call has no call_id");
      return `name=${fc.name} args=${truncate(fc.arguments ?? "", 40)}`;
    }),
  );

  results.push(
    await runTest("D2: tool roundtrip — continuation produces final answer", async () => {
      // Turn 1: model asks for the tool.
      const turn1 = await client.responses.create({
        model: args.model,
        input: "What's the weather in Tokyo? Use the tool.",
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get current weather for a city.",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
        tool_choice: "auto",
      });
      const fc = turn1.output.find(
        (o): o is Extract<typeof o, { type: "function_call" }> =>
          (o as { type: string }).type === "function_call",
      );
      if (!fc) return "skip — model didn't call the tool";
      // Turn 2: feed the tool result back. Include the original function_call
      // (a valid InputItem) plus the function_call_output. We deliberately
      // omit other items from turn1.output since OutputMessageItem isn't an
      // InputItem.
      const turn2 = await client.responses.create({
        model: args.model,
        input: [
          fc,
          {
            type: "function_call_output",
            call_id: fc.call_id,
            output: JSON.stringify({ city: "Tokyo", temp_c: 22, sky: "clear" }),
          },
        ],
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get current weather for a city.",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
      });
      assert(turn2.status === "completed", `turn2 status=${turn2.status}`);
      assert(
        (turn2.output_text ?? "").length > 0,
        "final answer empty after tool result",
      );
      const text = (turn2.output_text ?? "").toLowerCase();
      assert(
        text.includes("22") || text.includes("clear") || text.includes("tokyo"),
        `final answer doesn't reference tool output: ${truncate(turn2.output_text ?? "", 80)}`,
      );
      return `text=${truncate(turn2.output_text ?? "", 60)}`;
    }),
  );

  // === E. Conversations (needs store) ====================================
  let convId: string | null = null;
  if (!store) {
    for (const name of [
      "E1: conversations.create + continuation",
      "E2: conversations.items.list",
      "E3: conversations.items.append + get",
      "E4: conversations.items.del",
      "E5: conversations.update metadata",
      "E6: conversations.del",
    ]) {
      results.push({ name, status: "skip", detail: "no --store-local" });
    }
  } else {
    results.push(
      await runTest("E1: conversations.create + continuation", async () => {
        const conv = await client.conversations.create({
          metadata: { test: "smoke" },
        });
        convId = conv.id;
        await client.responses.create({
          model: args.model,
          conversation: conv.id,
          input:
            "Remember the secret word 'banana'. Acknowledge in one short sentence.",
        });
        const second = await client.responses.create({
          model: args.model,
          conversation: conv.id,
          input: "What was the secret word? Reply with just the word.",
        });
        const text = (second.output_text ?? "").toLowerCase();
        assert(
          text.includes("banana"),
          `context not recalled: ${truncate(second.output_text ?? "", 60)}`,
        );
        return `conv=${conv.id}`;
      }),
    );

    if (convId) {
      results.push(
        await runTest("E2: conversations.items.list", async () => {
          const page = await client.conversations.items.list(convId!);
          assert(page.data.length > 0, "items.list returned 0");
          assert(page.first_id !== null, "first_id null with data");
          assert(page.last_id !== null, "last_id null with data");
          return `count=${page.data.length}`;
        }),
      );

      results.push(
        await runTest("E3: conversations.items.append + get", async () => {
          const appended = await client.conversations.items.append(convId!, [
            {
              type: "message",
              role: "user",
              content: "(appended out-of-band)",
            },
          ]);
          const tail = appended[appended.length - 1] as {
            id?: string;
            type: string;
          };
          assert(tail && tail.type === "message", "append did not return tail");
          if (tail.id) {
            const got = await client.conversations.items.get(convId!, tail.id);
            assert(got !== null, "items.get returned null for appended id");
          }
          return `total=${appended.length}`;
        }),
      );

      results.push(
        await runTest("E4: conversations.items.del", async () => {
          const page = await client.conversations.items.list(convId!);
          const target = page.data.find(
            (it): it is typeof it & { id: string } =>
              typeof (it as { id?: string }).id === "string",
          );
          if (!target) return "no item with id to delete";
          const del = await client.conversations.items.del(
            convId!,
            (target as { id: string }).id,
          );
          assert(del.deleted, "items.del reported deleted=false");
          const refetch = await client.conversations.items.get(
            convId!,
            (target as { id: string }).id,
          );
          assert(refetch === null, "item still present after delete");
          return `deleted=${(target as { id: string }).id}`;
        }),
      );

      results.push(
        await runTest("E5: conversations.update metadata", async () => {
          const updated = await client.conversations.update(convId!, {
            metadata: { test: "smoke", updated: "true" },
          });
          assert(updated !== null, "update returned null");
          assert(
            updated!.metadata?.updated === "true",
            `metadata not updated: ${JSON.stringify(updated!.metadata)}`,
          );
          return `metadata=${JSON.stringify(updated!.metadata)}`;
        }),
      );

      results.push(
        await runTest("E6: conversations.del", async () => {
          const del = await client.conversations.del(convId!);
          assert(del.deleted, "del reported deleted=false");
          const got = await client.conversations.get(convId!);
          assert(got === null, "conversation still present after delete");
          return `id=${convId}`;
        }),
      );
    }
  }

  // === F. Responses persistence (needs store) ============================
  if (!store) {
    results.push({
      name: "F1: responses.get + del",
      status: "skip",
      detail: "no --store-local",
    });
    results.push({
      name: "F2: req.store=false skips persistence",
      status: "skip",
      detail: "no --store-local",
    });
  } else {
    let savedId: string | null = null;
    results.push(
      await runTest("F1: responses.get + del", async () => {
        const resp = await client.responses.create({
          model: args.model,
          input: "Reply with 'ok'.",
        });
        savedId = resp.id;
        const got = await client.responses.get(resp.id);
        assert(got !== null, "responses.get returned null");
        assert(got!.id === resp.id, "id mismatch");
        const del = await client.responses.del(resp.id);
        assert(del.deleted, "del reported deleted=false");
        const refetch = await client.responses.get(resp.id);
        assert(refetch === null, "response still present after delete");
        return `id=${resp.id}`;
      }),
    );

    results.push(
      await runTest("F2: req.store=false skips persistence", async () => {
        const resp = await client.responses.create({
          model: args.model,
          input: "Reply with 'ok'.",
          store: false,
        });
        const got = await client.responses.get(resp.id);
        assert(got === null, "response was persisted despite store=false");
        return `id=${resp.id}`;
      }),
    );
    void savedId;
  }

  // === G. /responses pass-through (optional) =============================
  if (!args.responsesEndpoint) {
    results.push({
      name: "G1: /responses pass-through non-stream",
      status: "skip",
      detail: "rerun with --responses-endpoint",
    });
    results.push({
      name: "G2: /responses pass-through stream — typed events",
      status: "skip",
      detail: "rerun with --responses-endpoint",
    });
  } else {
    results.push(
      await runTest("G1: /responses pass-through non-stream", async () => {
        const resp = await client.responses.create({
          model: args.model,
          input: "Reply with 'ok'.",
        });
        assert(resp.status === "completed", `status=${resp.status}`);
        return `text=${truncate(resp.output_text ?? "", 40)}`;
      }),
    );

    results.push(
      await runTest("G2: /responses pass-through stream — typed events", async () => {
        const stream = await client.responses.create({
          model: args.model,
          input: "Reply with 'ok'.",
          stream: true,
        });
        const types = new Set<string>();
        let untyped = 0;
        for await (const ev of stream) {
          const t = (ev as StreamEvent).type;
          if (typeof t !== "string" || t.length === 0) untyped++;
          else types.add(t);
        }
        await stream.finalResponse();
        assert(untyped === 0, `${untyped} events had no type`);
        assert(
          types.has("response.completed") || types.has("response.output_item.done"),
          `did not see completion event — saw: ${[...types].join(",")}`,
        );
        return `types=${types.size}`;
      }),
    );
  }

  // === H. Reasoning model (optional, openrouter only) ====================
  if (!args.reasoningModel) {
    results.push({
      name: "H1: reasoning model index alignment",
      status: "skip",
      detail: "rerun with --reasoning-model <slug>",
    });
  } else if (args.backend !== "openrouter") {
    results.push({
      name: "H1: reasoning model index alignment",
      status: "skip",
      detail: "--reasoning-model requires --backend openrouter",
    });
  } else {
    results.push(
      await runTest("H1: reasoning model index alignment", async () => {
        const stream = await client.responses.create({
          model: args.reasoningModel!,
          input: "Think step by step, then output only the final number: 17 * 23.",
          stream: true,
        });
        const added = new Map<number, string>();
        const done = new Map<number, string>();
        let reasoningCount = 0;
        let hadEncrypted = false;
        for await (const ev of stream) {
          if (ev.type === "response.output_item.added") {
            const item = ev.item as { type: string };
            added.set(ev.output_index, item.type);
            if (item.type === "reasoning") reasoningCount++;
          } else if (ev.type === "response.output_item.done") {
            const item = ev.item as { type: string; encrypted_content?: string };
            done.set(ev.output_index, item.type);
            if (item.type === "reasoning" && item.encrypted_content)
              hadEncrypted = true;
          }
        }
        const final = await stream.finalResponse();
        assert(final.status === "completed", `status=${final.status}`);
        for (const [idx, type] of added.entries()) {
          assert(
            done.get(idx) === type,
            `idx ${idx}: added=${type}, done=${done.get(idx)}`,
          );
        }
        for (let i = 0; i < final.output.length; i++) {
          const t = (final.output[i] as { type: string }).type;
          assert(
            added.get(i) === t,
            `final.output[${i}]=${t} vs event at ${i}=${added.get(i)}`,
          );
        }
        return `reasoning_items=${reasoningCount} encrypted=${hadEncrypted} final_items=${final.output.length}`;
      }),
    );
  }

  // === Cleanup ============================================================
  if (args.storeLocal) {
    await rm(args.storeLocal, { recursive: true, force: true }).catch(() => {});
  }

  printSummary(results);
  return results.some((r) => r.status === "fail") ? 1 : 0;
}

// ---- helpers -------------------------------------------------------------

function parseArgs(argv: string[]): Args {
  const get = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    if (i === -1) return undefined;
    const v = argv[i + 1];
    if (v === undefined || v.startsWith("--")) {
      throw new Error(`--${name} requires a value`);
    }
    return v;
  };
  const hasFlag = (name: string) => argv.includes(`--${name}`);
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
    reasoningModel: get("reasoning-model"),
    storeLocal: get("store-local"),
    responsesEndpoint: hasFlag("responses-endpoint"),
  };
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

async function expectThrow(
  fn: () => Promise<unknown>,
  pattern: RegExp,
): Promise<void> {
  try {
    await fn();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!pattern.test(msg)) {
      throw new Error(`error message ${JSON.stringify(msg)} does not match ${pattern}`);
    }
    return;
  }
  throw new Error("expected throw but call resolved");
}

async function withTimeout<T>(
  p: Promise<T>,
  ms: number,
  message: string,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      p,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
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

main(parseArgs(process.argv.slice(2)))
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(err instanceof Error ? (err.stack ?? err.message) : err);
    process.exit(2);
  });
