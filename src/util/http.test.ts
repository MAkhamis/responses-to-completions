import { afterEach, describe, expect, it, vi } from "vitest";
import {
  describeErrorBody,
  isRetryableFetchError,
  isRetryableResponse,
  retryDelayMs,
} from "./http.js";

const response = (status: number, headers: Record<string, string> = {}) => ({
  status,
  headers: new Headers(headers),
});

describe("describeErrorBody", () => {
  it("reduces a Cloudflare error page to its title and Ray ID", () => {
    const page = `<!DOCTYPE html>
<html class="no-js" lang="en-US"><head>
<title>api.openai.com | 504: Gateway time-out</title>
<style>body { color: red }</style>
</head><body>
<span class="cf-footer-item">Cloudflare Ray ID: <strong class="font-semibold">a43247a5be46f282</strong></span>
</body></html>`;

    expect(describeErrorBody(page)).toBe(
      "api.openai.com | 504: Gateway time-out (HTML error page, Cloudflare Ray ID a43247a5be46f282)",
    );
  });

  it("reads a plain proxy page, decoding its entities", () => {
    expect(
      describeErrorBody(
        "<html>\r\n<head><title>502 Bad Gateway &amp; friends&#39;</title></head>\r\n<body><center><h1>502 Bad Gateway</h1></center></body></html>",
      ),
    ).toBe("502 Bad Gateway & friends' (HTML error page)");
  });

  it("falls back to the page's text when it has no title", () => {
    expect(
      describeErrorBody(
        "<html><body><script>var x = '<b>';</script><h1>Service   Unavailable</h1></body></html>",
      ),
    ).toBe("Service Unavailable (HTML error page)");
  });

  it("keeps a JSON error verbatim", () => {
    const json =
      '{"error":{"message":"Conversation not found","type":"invalid_request_error"}}';
    expect(describeErrorBody(json)).toBe(json);
  });

  it("caps a long non-HTML body and says how long it was", () => {
    const long = "x".repeat(1500);
    expect(describeErrorBody(long, 10)).toBe("xxxxxxxxxx… (1500 chars)");
  });

  it("names an empty body instead of printing nothing", () => {
    expect(describeErrorBody("")).toBe("(empty body)");
    expect(describeErrorBody("  \n")).toBe("(empty body)");
  });
});

describe("isRetryableResponse", () => {
  it("treats 408, 409, 429 and every 5xx as transient", () => {
    for (const status of [408, 409, 429, 500, 502, 503, 504, 520, 524]) {
      expect(isRetryableResponse(response(status), true)).toBe(true);
    }
    for (const status of [400, 401, 403, 404, 422]) {
      expect(isRetryableResponse(response(status), true)).toBe(false);
    }
  });

  it("retries a request that must not run twice only when it cannot have landed", () => {
    expect(isRetryableResponse(response(429), false)).toBe(true);
    for (const status of [408, 409, 500, 502, 503, 504]) {
      expect(isRetryableResponse(response(status), false)).toBe(false);
    }
  });

  it("lets the server's x-should-retry hint decide", () => {
    expect(
      isRetryableResponse(response(503, { "x-should-retry": "false" }), true),
    ).toBe(false);
    expect(
      isRetryableResponse(response(400, { "x-should-retry": "true" }), false),
    ).toBe(true);
  });

  it("copes with a response that carries no headers object", () => {
    expect(isRetryableResponse({ status: 503 } as Response, true)).toBe(true);
  });
});

describe("isRetryableFetchError", () => {
  const failed = (code: string) =>
    new TypeError("fetch failed", {
      cause: Object.assign(new Error(code), { code }),
    });

  it("retries a repeatable request after any connection failure", () => {
    expect(isRetryableFetchError(failed("ECONNRESET"), undefined, true)).toBe(
      true,
    );
    expect(
      isRetryableFetchError(new TypeError("fetch failed"), undefined, true),
    ).toBe(true);
  });

  it("retries an unrepeatable request only when the connection never opened", () => {
    for (const code of [
      "ECONNREFUSED",
      "ENOTFOUND",
      "EAI_AGAIN",
      "UND_ERR_CONNECT_TIMEOUT",
    ]) {
      expect(isRetryableFetchError(failed(code), undefined, false)).toBe(true);
    }
    for (const code of [
      "ECONNRESET",
      "UND_ERR_SOCKET",
      "UND_ERR_HEADERS_TIMEOUT",
    ]) {
      expect(isRetryableFetchError(failed(code), undefined, false)).toBe(false);
    }
  });

  it("never retries once the caller has aborted", () => {
    const ac = new AbortController();
    ac.abort(new Error("done"));
    expect(isRetryableFetchError(failed("ECONNRESET"), ac.signal, true)).toBe(
      false,
    );
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    expect(isRetryableFetchError(abort, undefined, true)).toBe(false);
  });
});

describe("retryDelayMs", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("backs off from 0.5 s, doubling to an 8 s ceiling, with up to 25% jitter", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    expect([0, 1, 2, 3, 4, 5].map((n) => retryDelayMs(n))).toEqual([
      500, 1000, 2000, 4000, 8000, 8000,
    ]);
    vi.spyOn(Math, "random").mockReturnValue(1);
    expect(retryDelayMs(0)).toBe(375);
  });

  it("honors retry-after-ms, then retry-after in seconds or as a date", () => {
    expect(retryDelayMs(0, response(429, { "retry-after-ms": "1250" }))).toBe(
      1250,
    );
    expect(retryDelayMs(0, response(429, { "retry-after": "3" }))).toBe(3000);

    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-30T09:44:50Z"));
    expect(
      retryDelayMs(
        0,
        response(503, { "retry-after": "Wed, 30 Sep 2026 09:45:00 GMT" }),
      ),
    ).toBe(10_000);
  });

  it("ignores a hint over a minute, or one it cannot read, in favor of the backoff", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    expect(retryDelayMs(1, response(429, { "retry-after": "120" }))).toBe(1000);
    expect(retryDelayMs(1, response(429, { "retry-after": "soon" }))).toBe(
      1000,
    );
    expect(retryDelayMs(1, response(429, { "retry-after-ms": "0" }))).toBe(
      1000,
    );
  });
});
