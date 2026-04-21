import type { Response } from "express";

/**
 * Minimal SSE writer. Sends `event: <type>\ndata: <json>\n\n` framed events.
 * Sets appropriate headers and disables proxy buffering. Flushing is
 * implicit via Node's default write behavior.
 */
export class SseWriter {
  private closed = false;
  constructor(private res: Response) {
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();
  }

  send(event: string, data: unknown): void {
    if (this.closed) return;
    const payload = JSON.stringify(data);
    this.res.write(`event: ${event}\n`);
    // data can contain newlines — split into multiple data: lines per SSE spec
    for (const line of payload.split("\n")) {
      this.res.write(`data: ${line}\n`);
    }
    this.res.write("\n");
  }

  comment(text: string): void {
    if (this.closed) return;
    this.res.write(`: ${text}\n\n`);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.res.end();
  }
}
