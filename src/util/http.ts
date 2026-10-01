/**
 * Transport helpers for network-backed stores and adapters: which failures
 * are worth another attempt, how long to wait before it, and how to turn an
 * error body into something a person can read in a log line.
 */

const INITIAL_RETRY_DELAY_MS = 500;
const MAX_RETRY_DELAY_MS = 8_000;
/** A server hint beyond this is ignored in favor of the default backoff. */
const MAX_RETRY_AFTER_MS = 60_000;

/**
 * Whether an error response is worth sending the request again. Mirrors the
 * rules of OpenAI's own SDK: the server's `x-should-retry` hint wins, then
 * 408/409/429 and every 5xx — including the 52x pages Cloudflare serves in
 * front of a provider — are transient.
 *
 * `repeatable: false` is for a request that must not run twice. A 5xx from a
 * gateway says nothing about whether the origin applied the request, so such
 * a request is retried only when the answer proves it was not: a 429, or an
 * explicit `x-should-retry: true`.
 */
export function isRetryableResponse(
  res: Pick<Response, "status" | "headers">,
  repeatable: boolean,
): boolean {
  const hint = res.headers?.get?.("x-should-retry");
  if (hint === "true") return true;
  if (hint === "false") return false;
  if (res.status === 429) return true;
  if (!repeatable) return false;
  return res.status === 408 || res.status === 409 || res.status >= 500;
}

/**
 * Whether a rejected `fetch` is worth another attempt. Never once the caller
 * has aborted — a retry would outlive the request it belongs to. A request
 * that must not run twice is retried only when the connection never opened,
 * so the server cannot have seen it.
 */
export function isRetryableFetchError(
  err: unknown,
  signal: AbortSignal | undefined,
  repeatable: boolean,
): boolean {
  if (signal?.aborted) return false;
  if ((err as { name?: unknown } | null)?.name === "AbortError") return false;
  return repeatable || neverConnected(err);
}

/** Codes Node and undici use for a connection that was never established. */
const CONNECT_ERROR_CODES = new Set([
  "ECONNREFUSED",
  "ENOTFOUND",
  "EAI_AGAIN",
  "ENETUNREACH",
  "EHOSTUNREACH",
  "UND_ERR_CONNECT_TIMEOUT",
]);

/** undici wraps the socket error as `cause` of a `TypeError: fetch failed`. */
function neverConnected(err: unknown): boolean {
  let current: unknown = err;
  for (let depth = 0; current && depth < 4; depth++) {
    const code = (current as { code?: unknown }).code;
    if (typeof code === "string" && CONNECT_ERROR_CODES.has(code)) return true;
    current = (current as { cause?: unknown }).cause;
  }
  return false;
}

/**
 * How long to wait before retry number `retry` (0-based): the server's
 * `retry-after-ms` / `retry-after` when it gives a usable one (up to a
 * minute), otherwise 0.5 s doubling to 8 s, less up to 25% jitter.
 */
export function retryDelayMs(
  retry: number,
  res?: Pick<Response, "headers">,
): number {
  const hinted = retryAfterMs(res);
  if (hinted !== undefined && hinted > 0 && hinted <= MAX_RETRY_AFTER_MS) {
    return hinted;
  }
  const base = Math.min(
    INITIAL_RETRY_DELAY_MS * 2 ** retry,
    MAX_RETRY_DELAY_MS,
  );
  return base * (1 - Math.random() * 0.25);
}

function retryAfterMs(res?: Pick<Response, "headers">): number | undefined {
  const headers = res?.headers;
  if (!headers?.get) return undefined;
  const ms = Number.parseFloat(headers.get("retry-after-ms") ?? "");
  if (!Number.isNaN(ms)) return ms;
  const after = headers.get("retry-after");
  if (!after) return undefined;
  const seconds = Number.parseFloat(after);
  if (!Number.isNaN(seconds)) return seconds * 1000;
  const date = Date.parse(after);
  return Number.isNaN(date) ? undefined : date - Date.now();
}

/** Waits `ms`, or rejects with the signal's reason as soon as it aborts. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * The signal for one attempt of a request: it follows the caller's signal and
 * also fires after `timeoutMs` without an answer.
 *
 * - `settle()` stops the clock once the response has started — a large body
 *   still downloading is not a hung server.
 * - `release()` detaches from the caller's signal when the attempt is over,
 *   so a long-lived signal shared by many requests (every page of a listing)
 *   does not collect one listener per request.
 */
export function attemptSignal(
  parent: AbortSignal | undefined,
  timeoutMs: number,
): {
  signal: AbortSignal;
  timedOut(): boolean;
  settle(): void;
  release(): void;
} {
  const controller = new AbortController();
  let timedOut = false;
  const follow = () => controller.abort(parent?.reason);
  if (parent?.aborted) controller.abort(parent.reason);
  else parent?.addEventListener("abort", follow, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    const reason = new Error(`no response within ${timeoutMs} ms`);
    reason.name = "TimeoutError";
    controller.abort(reason);
  }, timeoutMs);
  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    settle: () => clearTimeout(timer),
    release: () => {
      clearTimeout(timer);
      parent?.removeEventListener("abort", follow);
    },
  };
}

/** Frees the connection behind a response whose body will not be read. */
export async function discardBody(res: Response): Promise<void> {
  try {
    await res.body?.cancel();
  } catch {
    // The connection is being dropped either way.
  }
}

/**
 * An error body as a one-line reading for an error message. Gateways in
 * front of an API (Cloudflare, load balancers) answer a 5xx with a whole HTML
 * page — kilobytes of markup that bury the one useful line. Such a page is
 * reduced to its `<title>`, plus Cloudflare's Ray ID when present (the id a
 * provider's support asks for). Anything else is kept verbatim up to
 * `maxLength` characters.
 */
export function describeErrorBody(body: string, maxLength = 1000): string {
  const text = body.trim();
  if (!text) return "(empty body)";
  if (/^<(?:!doctype\s+html|html[\s>]|head[\s>]|body[\s>]|!--)/i.test(text)) {
    return describeHtmlPage(text);
  }
  return text.length <= maxLength
    ? text
    : `${text.slice(0, maxLength)}… (${text.length} chars)`;
}

function describeHtmlPage(html: string): string {
  const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1];
  const rayId = /Ray ID:?\s*(?:<[^>]*>\s*)*([0-9a-f]{8,})/i.exec(html)?.[1];
  const heading =
    readableText(title ?? html).slice(0, 200) || "(no readable text)";
  return `${heading} (HTML error page${rayId ? `, Cloudflare Ray ID ${rayId}` : ""})`;
}

function readableText(html: string): string {
  return decodeEntities(
    html
      .replace(/<(script|style)\b[\s\S]*?<\/\1\s*>/gi, " ")
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<[^>]*>/g, " "),
  )
    .replace(/\s+/g, " ")
    .trim();
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  bull: "•",
};

function decodeEntities(text: string): string {
  return text.replace(
    /&(#x[0-9a-f]+|#\d+|[a-z]+);/gi,
    (entity, name: string) => {
      if (name[0] !== "#") return NAMED_ENTITIES[name.toLowerCase()] ?? entity;
      const hex = name[1] === "x" || name[1] === "X";
      const code = Number.parseInt(name.slice(hex ? 2 : 1), hex ? 16 : 10);
      return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : entity;
    },
  );
}
