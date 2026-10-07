/**
 * Audio transcription (speech-to-text) shapes, normalized across providers.
 * https://platform.openai.com/docs/api-reference/audio/createTranscription
 * https://openrouter.ai/docs/guides/overview/multimodal/stt
 */

/** Raw audio bytes. A `Blob` keeps its own MIME type unless `mime_type` overrides it. */
export type TranscriptionAudio = Blob | ArrayBuffer | Uint8Array;

/**
 * OpenRouter's routing controls for a transcription. Only the data-policy
 * keys apply to speech-to-text — `order`/`only`/`ignore` are not honored on
 * that endpoint.
 */
export interface OpenRouterTranscriptionProvider {
  /** Route only to zero-data-retention endpoints. */
  zdr?: boolean;
  /** Whether the provider may use the audio for training. */
  data_collection?: "allow" | "deny";
  /**
   * Provider-specific parameters, keyed by the endpoint tag from
   * `/models/{slug}/endpoints`, sent under the provider's own field names
   * (e.g. a vocabulary list). Keys a provider does not know are dropped.
   */
  options?: Record<string, Record<string, unknown>>;
}

export interface TranscriptionRequest {
  /**
   * Transcription model id — e.g. "gpt-transcribe" on OpenAI or
   * "google/gemini-3.5-transcribe" on OpenRouter. An adapter's `forceModel`
   * is NOT applied: it names a chat model.
   */
  model: string;
  /** The recording. */
  file: TranscriptionAudio;
  /**
   * File name with its extension ("note.webm") — providers identify the
   * audio format from it.
   */
  filename: string;
  /** MIME type of `file`, e.g. "audio/webm". Inferred from the extension when omitted. */
  mime_type?: string;
  /** One expected language, ISO-639-1 (e.g. "ar"). */
  language?: string;
  /**
   * Several expected languages — OpenAI's `gpt-transcribe` takes them as
   * `languages[]` for code-switched speech. Wins over `language`. OpenRouter
   * takes a single code: one entry is sent as `language`, more than one is
   * left out so the provider detects the language itself.
   */
  languages?: string[];
  /** Free-form context about the recording. Not forwarded by OpenRouter. */
  prompt?: string;
  /**
   * Literal terms expected in the audio (OpenAI `gpt-transcribe`). Not
   * forwarded by OpenRouter — pass a provider's own vocabulary field through
   * `provider.options` instead.
   */
  keywords?: string[];
  /** Sampling temperature, 0–1. */
  temperature?: number;
  /** "json" (the provider default) or "verbose_json". */
  response_format?: "json" | "verbose_json";
  /** OpenRouter only — data-policy routing and provider-specific options. */
  provider?: OpenRouterTranscriptionProvider;
}

/**
 * What the request consumed. Token-billed models report tokens,
 * duration-billed ones report `seconds`; OpenRouter reports `seconds` plus
 * the charged `cost` for every model.
 */
export interface TranscriptionUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_tokens_details?: {
    audio_tokens?: number;
    text_tokens?: number;
  };
  /** Audio duration in seconds. */
  seconds?: number;
  /** Total charged for the request, in USD (OpenRouter). */
  cost?: number;
}

export interface TranscriptionResponse {
  text: string;
  /** The model that was asked for. */
  model: string;
  /**
   * Languages the model detected, as codes. Present only when the model
   * reports them (OpenAI `gpt-transcribe`); an empty array means none could
   * be detected reliably.
   */
  languages?: string[];
  /** Detected or requested language (verbose_json). */
  language?: string;
  /** Audio duration in seconds (verbose_json). */
  duration?: number;
  usage: TranscriptionUsage | null;
  /** Provider request id — OpenAI `x-request-id`, OpenRouter `x-generation-id`. */
  request_id?: string;
}
