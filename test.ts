import { ResponsesClient, StorePersistenceError } from "./src/client.js";
import type { Store } from "./src/store/store.js";
import type { StreamEvent } from "./src/translate/stream.js";

/**
 * Run with: npx tsx test.ts
 * Needs OPENAI_API_KEY in the environment.
 */

const AIClient = new ResponsesClient({
  source: "openAI",
  config: {
    // An empty key throws at construction — the source needs real credentials.
    apiKey: process.env.BACKEND_API_KEY ?? process.env.OPENAI_API_KEY!,
    endpoint: "responses",
  },
  store: true,
  store_client: "S3",
  store_config: {
    bucket: process.env.STORE_S3_BUCKET as string,
    clientConfig: {
      region: process.env.STORE_S3_REGION,
      credentials: {
        accessKeyId: process.env.STORE_S3_ACCESS_KEY_ID as string,
        secretAccessKey: process.env.STORE_S3_SECRET_ACCESS_KEY as string,
      },
    },
  },
});

// 1. Plain call — `input` is a string, the reply is on `output_text`.
async function basic() {
  const result = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: "Give me one sentence about the Dead Sea.",
    stream: false,
  });

  console.log(result.output_text);
  console.log(result.usage); // { input_tokens, output_tokens, total_tokens, ... }
}

// 2. Instructions + sampling knobs.
async function withInstructions() {
  const result = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    instructions: "You are terse. Answer in at most 10 words.",
    input: "What is an SDK?",
    // temperature: 0.2,
    max_output_tokens: 100,
    stream: false,
  });

  console.log(result.output_text);
}

// 3. Multi-turn input — roles instead of a bare string.
async function multiTurn() {
  const result = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: [
      { type: "message", role: "user", content: "We are in London." },
      { type: "message", role: "assistant", content: "Nice city" },
      { type: "message", role: "user", content: "Where are we ?" },
    ],
    stream: false,
  });

  console.log(result.output_text); 
}

// 4. Streaming — `stream: true` returns an async-iterable, not a response.
async function streaming() {
  const stream = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: "Count from 1 to 5.",
    stream: true,
  });

  for await (const event of stream) {
    if (event.type === "response.output_text.delta") {
      process.stdout.write(event.delta);
    }
  }

  const final = await stream.finalResponse();
  console.log("\nid:", final.id, "status:", final.status);
}

// 5. Function calling — the model answers with a `function_call` item you run
//    yourself, then feed the output back on the next turn.
async function withTools() {
  const result = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: "What is the weather in Amman?",
    tools: [
      {
        type: "function",
        name: "get_weather",
        description: "Current weather for a city.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    ],
    tool_choice: "auto",
    stream: false,
  });

  for (const item of result.output) {
    if (item.type === "function_call") {
      console.log(item.name, item.arguments); // "get_weather" '{"city":"Amman"}'
    }
  }
}

// 6. JSON output with a schema.
async function structured() {
  const result = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: "Extract the city and country: 'I flew to Amman, Jordan.'",
    text: {
      format: {
        type: "json_schema",
        name: "location",
        strict: true,
        schema: {
          type: "object",
          properties: { city: { type: "string" }, country: { type: "string" } },
          required: ["city", "country"],
          additionalProperties: false,
        },
      },
    },
    stream: false,
  });

  console.log(JSON.parse(result.output_text ?? "{}"));
}

// 7. Conversations — the store keeps the transcript, so each turn only sends
//    the new input. Needs `store: true` + a `store_client` on the client.
async function conversation() {
  const conv = await AIClient.conversations.create({
    metadata: { user: "u_42" },
  });

  await AIClient.responses.create({
    model: "gpt-5.6-luna",
    conversation: conv.id,
    input: "Remember the secret word 'banana'.",
    stream: false,
  });

  const second = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    conversation: conv.id,
    input: "What was the secret word?",
    stream: false,
  });

  console.log(second.output_text); // "banana"

  // Everything persisted so far, as canonical Responses-API items.
  const { data } = await AIClient.conversations.items.list(conv.id);
  console.log(data.length, "items");

  // Append items yourself (no model call), then clean up.
  await AIClient.conversations.items.append(conv.id, [
    { type: "message", role: "user", content: "Noted." },
  ]);
  await AIClient.conversations.del(conv.id);
}

// 8. Same continuity without a conversation id — chain by response id.
async function previousResponse() {
  const first = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    input: "My favourite colour is teal.",
    stream: false,
  });

  const second = await AIClient.responses.create({
    model: "gpt-5.6-luna",
    previous_response_id: first.id,
    input: "What is my favourite colour?",
    stream: false,
  });

  console.log(second.output_text); // "teal"
}

// 9. Store failures don't destroy a finished turn.
//
//    The model call and the store write are separate failure domains: by the
//    time the write runs the turn is already done and billed, and its output
//    exists nowhere else. So a store that refuses the write raises a
//    `StorePersistenceError` carrying the *completed* response, rather than
//    reporting the turn itself as failed with an empty output.
//
//    The model call below is real; only the store is rigged to fail. Reads
//    still work, so nothing else about the client changes.
const brokenStore: Store = {
  createConversation: async () => ({
    id: "conv_broken",
    object: "conversation",
    created_at: Math.floor(Date.now() / 1000),
    metadata: null,
  }),
  getConversation: async () => null,
  updateConversation: async () => null,
  deleteConversation: async () => ({ id: "conv_broken", deleted: true }),
  listItems: async () => ({ items: [], hasMore: false }),
  getItem: async () => null,
  deleteItem: async () => ({ id: "x", deleted: true }),
  getResponse: async () => null,
  deleteResponse: async () => ({ id: "x", deleted: true }),
  // The two writes `responses.create` performs. Either one failing takes the
  // same path through the client.
  appendItems: async () => {
    throw new Error("S3 503 SlowDown");
  },
  saveResponse: async () => {
    throw new Error("S3 503 SlowDown");
  },
};

async function storeFailureKeepsTheTurn() {
  // `createStore` runs inside the base constructor, so it must not read
  // subclass fields — `brokenStore` is module-level for that reason.
  class BrokenStoreClient extends ResponsesClient {
    protected override createStore(): Store {
      return brokenStore;
    }
  }

  const client = new BrokenStoreClient({
    source: "openAI",
    config: {
      apiKey: process.env.BACKEND_API_KEY ?? process.env.OPENAI_API_KEY!,
      endpoint: "responses",
    },
    store: true,
    store_client: "S3",
    // Overridden by createStore above, but the type still wants a target.
    store_config: { bucket: "unused" },
  });

  // --- non-streaming: the output comes back on the error ---
  let recovered: string | undefined;
  try {
    const result = await client.responses.create({
      model: "gpt-5.6-luna",
      input: "Say the word 'Hello World' and nothing else.",
      stream: false,
    });
    throw new Error(
      `expected a StorePersistenceError, got a clean response: ${result.id}`,
    );
  } catch (err) {
    if (!(err instanceof StorePersistenceError)) throw err;

    // Completed, not failed — nothing went wrong with the response itself.
    if (err.response.status !== "completed") {
      throw new Error(`expected status "completed", got "${err.response.status}"`);
    }
    if (!err.response.output_text) {
      throw new Error("the completed turn came back with no output");
    }
    recovered = err.response.output_text;

    console.log("non-streaming:");
    console.log("  message:  ", err.message);
    console.log("  cause:    ", (err.cause as Error)?.message);
    console.log("  status:   ", err.response.status);
    console.log("  recovered:", JSON.stringify(recovered));
    console.log("  usage:    ", err.response.usage);
  }

  // --- streaming: every event still goes out, only the promise rejects ---
  const stream = await client.responses.create({
    model: "gpt-5.6-luna",
    input: "Count from 1 to 3. but make delay 3s between them",
    stream: true,
  });

  // The events are the subject here, not the text they add up to:
  // `output_text` is a convenience view computed at the end, so reading it
  // would say nothing about what the stream actually delivered.
  const events: StreamEvent[] = [];
  for await (const event of stream) {
    events.push(event);
  }

  const types = events.map((e) => e.type);

  // The consumer was told the turn completed, because on the wire it did.
  if (types.at(-1) !== "response.completed") {
    throw new Error(`expected the stream to end on completed, got ${types.at(-1)}`);
  }
  if (types.includes("response.failed")) {
    throw new Error("a delivered turn was restated as failed");
  }

  // The failure surfaces here alone — this promise covers persistence too.
  const rejection = await stream.finalResponse().then(
    () => null,
    (e: unknown) => e,
  );
  if (!(rejection instanceof StorePersistenceError)) {
    throw new Error("finalResponse() did not report the store failure");
  }

  console.log("streaming:");
  console.log("  events:  ", events.length);
  for (const event of events) {
    console.log("   ", event.sequence_number, event.type, " : ", event.type == "response.output_text.delta" ? event.delta : null );
  }
  console.log("  rejected:", rejection.message);

  return events;
}

const main = async () => {
  await basic();
  await withInstructions();
  await multiTurn();
  await streaming();
  await withTools();
  await structured();
  await conversation();
  await previousResponse();
  await storeFailureKeepsTheTurn();
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
