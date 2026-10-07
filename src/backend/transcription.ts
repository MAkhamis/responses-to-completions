import type {
  TranscriptionRequest,
  TranscriptionResponse,
  TranscriptionUsage,
} from "../types/audio.js";

/**
 * Shared halves of the adapters' `transcribe()`: working out what format a
 * recording is in, turning its bytes into what each wire format wants, and
 * normalizing the providers' answers into one `TranscriptionResponse`.
 */

/**
 * Audio formats by file extension. `format` is the name OpenRouter's
 * `input_audio.format` accepts (wav, mp3, flac, m4a, ogg, webm, aac), so MP4
 * audio maps to "m4a" and the MPEG aliases to "mp3".
 */
const EXTENSION_FORMATS: Record<string, { format: string; mime: string }> = {
  webm: { format: "webm", mime: "audio/webm" },
  weba: { format: "webm", mime: "audio/webm" },
  m4a: { format: "m4a", mime: "audio/mp4" },
  mp4: { format: "m4a", mime: "audio/mp4" },
  aac: { format: "aac", mime: "audio/aac" },
  mp3: { format: "mp3", mime: "audio/mpeg" },
  mpeg: { format: "mp3", mime: "audio/mpeg" },
  mpga: { format: "mp3", mime: "audio/mpeg" },
  wav: { format: "wav", mime: "audio/wav" },
  ogg: { format: "ogg", mime: "audio/ogg" },
  oga: { format: "ogg", mime: "audio/ogg" },
  opus: { format: "ogg", mime: "audio/ogg" },
  flac: { format: "flac", mime: "audio/flac" },
};

const MIME_EXTENSIONS: Record<string, string> = {
  "audio/webm": "webm",
  "video/webm": "webm",
  "audio/mp4": "m4a",
  "audio/m4a": "m4a",
  "audio/x-m4a": "m4a",
  "video/mp4": "mp4",
  "audio/aac": "aac",
  "audio/x-aac": "aac",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/mpga": "mp3",
  "audio/wav": "wav",
  "audio/wave": "wav",
  "audio/x-wav": "wav",
  "audio/vnd.wave": "wav",
  "audio/ogg": "ogg",
  "audio/opus": "opus",
  "audio/flac": "flac",
  "audio/x-flac": "flac",
};

export interface DescribedAudio {
  /** The file name to send — given one, with an extension added when it had none. */
  filename: string;
  /** OpenRouter `input_audio.format`. */
  format: string;
  /** MIME type for the multipart part. */
  mime: string;
}

const extensionOf = (filename: string): string => {
  const match = /\.([A-Za-z0-9]+)$/.exec(filename.trim());
  return match ? match[1].toLowerCase() : "";
};

const baseMime = (mime: string | undefined): string =>
  (mime ?? "").split(";")[0].trim().toLowerCase();

/**
 * Which format a recording is in, from its file extension first (that is
 * what providers go by), then its MIME type. A name without a known
 * extension gets one from the MIME type, so the provider can still tell.
 */
export function describeAudio(req: TranscriptionRequest): DescribedAudio {
  const mime =
    baseMime(req.mime_type) ||
    baseMime(req.file instanceof Blob ? req.file.type : undefined);
  const byExtension = EXTENSION_FORMATS[extensionOf(req.filename)];
  if (byExtension) {
    return {
      filename: req.filename,
      format: byExtension.format,
      mime: mime || byExtension.mime,
    };
  }
  const extension = MIME_EXTENSIONS[mime];
  if (extension) {
    const byMime = EXTENSION_FORMATS[extension];
    const stem = req.filename.trim() || "audio";
    return {
      filename: `${stem}.${extension}`,
      format: byMime.format,
      mime,
    };
  }
  throw new Error(
    `transcription: cannot tell the audio format of "${req.filename}"${
      mime ? ` (${mime})` : ""
    } — give the file name an extension such as .webm, .m4a, .mp3, .wav, .ogg or .flac, or pass \`mime_type\`.`,
  );
}

/** The recording as a Blob of the given type. */
export async function audioBlob(
  req: TranscriptionRequest,
  mime: string,
): Promise<Blob> {
  const bytes =
    req.file instanceof Blob
      ? new Uint8Array(await req.file.arrayBuffer())
      : req.file instanceof ArrayBuffer
      ? new Uint8Array(req.file)
      : new Uint8Array(req.file);
  if (bytes.byteLength === 0) {
    throw new Error(`transcription: "${req.filename}" is empty`);
  }
  return new Blob([bytes], { type: mime });
}

/** The recording as raw base64 (no data-URI prefix). */
export async function audioBase64(req: TranscriptionRequest): Promise<string> {
  const blob = await audioBlob(req, "application/octet-stream");
  return Buffer.from(await blob.arrayBuffer()).toString("base64");
}

const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

/**
 * One usage shape for every provider. OpenAI sends `{type: "tokens", …,
 * input_token_details}` (note the singular "token") or `{type: "duration",
 * seconds}`; OpenRouter sends `seconds`, token counts and `cost` together.
 */
export function normalizeTranscriptionUsage(
  raw: unknown,
): TranscriptionUsage | null {
  if (!raw || typeof raw !== "object") return null;
  const usage = raw as Record<string, unknown>;
  const out: TranscriptionUsage = {};
  for (const key of [
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "seconds",
    "cost",
  ] as const) {
    const value = usage[key];
    if (isNumber(value)) out[key] = value;
  }
  const details = (usage.input_tokens_details ??
    usage.input_token_details) as Record<string, unknown> | undefined;
  if (details && typeof details === "object") {
    const normalized: NonNullable<TranscriptionUsage["input_tokens_details"]> =
      {};
    if (isNumber(details.audio_tokens)) {
      normalized.audio_tokens = details.audio_tokens;
    }
    if (isNumber(details.text_tokens)) {
      normalized.text_tokens = details.text_tokens;
    }
    if (Object.keys(normalized).length) out.input_tokens_details = normalized;
  }
  return out;
}

/** `gpt-transcribe` reports `[{ code: "ar" }]`; keep the codes. */
const normalizeLanguages = (raw: unknown): string[] | undefined => {
  if (!Array.isArray(raw)) return undefined;
  return raw
    .map((entry) =>
      typeof entry === "string"
        ? entry
        : entry && typeof entry === "object"
        ? (entry as { code?: unknown }).code
        : undefined,
    )
    .filter((code): code is string => typeof code === "string" && !!code);
};

export function normalizeTranscription(
  raw: unknown,
  model: string,
  requestId?: string | null,
): TranscriptionResponse {
  const body = (raw && typeof raw === "object" ? raw : {}) as Record<
    string,
    unknown
  >;
  const out: TranscriptionResponse = {
    text: typeof body.text === "string" ? body.text : "",
    model,
    usage: normalizeTranscriptionUsage(body.usage),
  };
  const languages = normalizeLanguages(body.languages);
  if (languages) out.languages = languages;
  if (typeof body.language === "string") out.language = body.language;
  if (isNumber(body.duration)) out.duration = body.duration;
  if (requestId) out.request_id = requestId;
  return out;
}
