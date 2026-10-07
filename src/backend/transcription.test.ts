import { describe, expect, it } from "vitest";
import { OpenAICompatAdapter, BackendError } from "./openai-compat.js";
import { OpenRouterAdapter } from "./open-router-compat.js";
import {
  describeAudio,
  normalizeTranscription,
  normalizeTranscriptionUsage,
} from "./transcription.js";

const AUDIO = new Uint8Array([1, 2, 3, 4, 5]);

type Captured = {
  url: string;
  headers: Record<string, string>;
  body: unknown;
};

const captureFetch = (
  calls: Captured[],
  json: unknown,
  responseHeaders: Record<string, string> = {},
  status = 200,
): typeof fetch =>
  (async (url: unknown, init?: RequestInit) => {
    calls.push({
      url: String(url),
      headers: (init?.headers ?? {}) as Record<string, string>,
      body: init?.body,
    });
    return new Response(
      typeof json === "string" ? json : JSON.stringify(json),
      { status, headers: responseHeaders },
    );
  }) as typeof fetch;

describe("OpenAICompatAdapter.transcribe", () => {
  it("posts multipart form data with gpt-transcribe hints", async () => {
    const calls: Captured[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.test/v1/",
      apiKey: "sk-x",
      forceModel: "chat-model",
      fetch: captureFetch(calls, {
        text: "مرحبا hello",
        languages: [{ code: "ar" }, { code: "en" }],
        usage: { type: "duration", seconds: 4.2 },
      }),
    });

    const result = await adapter.transcribe({
      model: "gpt-transcribe",
      file: AUDIO,
      filename: "note.webm",
      mime_type: "audio/webm;codecs=opus",
      languages: ["ar", "en"],
      language: "ar",
      prompt: "A sales voice note.",
      keywords: ["Pepsi", "SKU-42"],
    });

    expect(calls[0].url).toBe("https://api.openai.test/v1/audio/transcriptions");
    expect(calls[0].headers.authorization).toBe("Bearer sk-x");
    expect(
      Object.keys(calls[0].headers).map((key) => key.toLowerCase()),
    ).not.toContain("content-type");
    const form = calls[0].body as FormData;
    expect(form).toBeInstanceOf(FormData);
    // forceModel names a chat model and never reaches a transcription.
    expect(form.get("model")).toBe("gpt-transcribe");
    expect(form.getAll("languages[]")).toEqual(["ar", "en"]);
    expect(form.get("language")).toBeNull();
    expect(form.get("prompt")).toBe("A sales voice note.");
    expect(form.getAll("keywords[]")).toEqual(["Pepsi", "SKU-42"]);
    const file = form.get("file") as File;
    expect(file.name).toBe("note.webm");
    expect(file.type).toBe("audio/webm");
    expect(new Uint8Array(await file.arrayBuffer())).toEqual(AUDIO);

    expect(result).toEqual({
      text: "مرحبا hello",
      model: "gpt-transcribe",
      languages: ["ar", "en"],
      usage: { seconds: 4.2 },
    });
  });

  it("sends a single language and adds the extension the MIME type implies", async () => {
    const calls: Captured[] = [];
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.test/v1",
      headers: { "Content-Type": "application/json", "x-team": "t1" },
      fetch: captureFetch(
        calls,
        {
          text: "hi",
          usage: {
            type: "tokens",
            input_tokens: 120,
            output_tokens: 8,
            total_tokens: 128,
            input_token_details: { audio_tokens: 110, text_tokens: 10 },
          },
        },
        { "x-request-id": "req_123" },
      ),
    });

    const result = await adapter.transcribe({
      model: "gpt-4o-mini-transcribe",
      file: new Blob([AUDIO], { type: "audio/mp4" }),
      filename: "recording",
      language: "en",
      temperature: 0,
    });

    const form = calls[0].body as FormData;
    expect(form.get("language")).toBe("en");
    expect(form.getAll("languages[]")).toEqual([]);
    expect(form.get("temperature")).toBe("0");
    expect((form.get("file") as File).name).toBe("recording.m4a");
    // A configured JSON content-type would hide the multipart boundary.
    expect(calls[0].headers).toEqual({ "x-team": "t1" });
    expect(result.usage).toEqual({
      input_tokens: 120,
      output_tokens: 8,
      total_tokens: 128,
      input_tokens_details: { audio_tokens: 110, text_tokens: 10 },
    });
    expect(result.request_id).toBe("req_123");
    expect(result).not.toHaveProperty("languages");
  });

  it("throws BackendError on a non-2xx answer", async () => {
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.test/v1",
      fetch: captureFetch([], '{"error":{"message":"bad audio"}}', {}, 400),
    });

    const failure = await adapter
      .transcribe({ model: "gpt-transcribe", file: AUDIO, filename: "a.mp3" })
      .catch((error) => error);

    expect(failure).toBeInstanceOf(BackendError);
    expect(failure.status).toBe(400);
  });
});

describe("OpenRouterAdapter.transcribe", () => {
  it("inlines the audio as base64 JSON with one language and the provider policy", async () => {
    const calls: Captured[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-x",
      forceModel: "chat-model",
      provider: { order: ["Anthropic"] },
      fetch: captureFetch(
        calls,
        {
          text: "hello",
          usage: {
            seconds: 9.2,
            input_tokens: 83,
            output_tokens: 30,
            total_tokens: 113,
            cost: 0.000508,
          },
        },
        { "x-generation-id": "gen-1" },
      ),
    });

    const result = await adapter.transcribe({
      model: "google/gemini-3.5-transcribe",
      file: AUDIO,
      filename: "note.mp4",
      languages: ["ar"],
      prompt: "ignored",
      keywords: ["ignored"],
      provider: { zdr: true, options: { azure: { phraseList: ["x"] } } },
    });

    expect(calls[0].url).toBe("https://openrouter.ai/api/v1/audio/transcriptions");
    expect(calls[0].headers["content-type"]).toBe("application/json");
    expect(JSON.parse(String(calls[0].body))).toEqual({
      model: "google/gemini-3.5-transcribe",
      input_audio: {
        data: Buffer.from(AUDIO).toString("base64"),
        format: "m4a",
      },
      language: "ar",
      provider: { zdr: true, options: { azure: { phraseList: ["x"] } } },
    });
    expect(result).toEqual({
      text: "hello",
      model: "google/gemini-3.5-transcribe",
      usage: {
        seconds: 9.2,
        input_tokens: 83,
        output_tokens: 30,
        total_tokens: 113,
        cost: 0.000508,
      },
      request_id: "gen-1",
    });
  });

  it("leaves the language out when several are expected", async () => {
    const calls: Captured[] = [];
    const adapter = new OpenRouterAdapter({
      apiKey: "or-x",
      fetch: captureFetch(calls, { text: "", usage: { seconds: 1, cost: 0 } }),
    });

    await adapter.transcribe({
      model: "openai/whisper-1",
      file: AUDIO,
      filename: "note.ogg",
      languages: ["ar", "en"],
      language: "ar",
    });

    const body = JSON.parse(String(calls[0].body));
    expect(body).not.toHaveProperty("language");
    expect(body.input_audio.format).toBe("ogg");
  });
});

describe("transcription helpers", () => {
  it("prefers the extension and strips MIME parameters", () => {
    expect(
      describeAudio({
        model: "m",
        file: AUDIO,
        filename: "x.WAV",
        mime_type: "audio/wav; rate=16000",
      }),
    ).toEqual({ filename: "x.WAV", format: "wav", mime: "audio/wav" });
  });

  it("refuses audio whose format cannot be told", () => {
    expect(() =>
      describeAudio({ model: "m", file: AUDIO, filename: "voice.bin" }),
    ).toThrow(/cannot tell the audio format/);
  });

  it("refuses an empty recording", async () => {
    const adapter = new OpenAICompatAdapter({
      baseUrl: "https://api.openai.test/v1",
      fetch: captureFetch([], { text: "" }),
    });
    await expect(
      adapter.transcribe({
        model: "gpt-transcribe",
        file: new Uint8Array(),
        filename: "a.webm",
      }),
    ).rejects.toThrow(/is empty/);
  });

  it("normalizes missing or odd usage and language shapes", () => {
    expect(normalizeTranscriptionUsage(undefined)).toBeNull();
    expect(normalizeTranscriptionUsage({ seconds: "3", cost: 0.1 })).toEqual({
      cost: 0.1,
    });
    expect(
      normalizeTranscription(
        { text: 42, languages: ["fr", { code: "en" }, {}, null] },
        "m",
      ),
    ).toEqual({ text: "", model: "m", languages: ["fr", "en"], usage: null });
    expect(
      normalizeTranscription({ text: "x", languages: [] }, "m").languages,
    ).toEqual([]);
  });
});
