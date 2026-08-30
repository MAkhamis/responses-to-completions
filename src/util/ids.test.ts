import { describe, expect, it } from "vitest";
import { genConvId, generateUUID } from "./ids.js";

describe("generateUUID", () => {
  it("emits base36 characters only, prefixed with the separator", () => {
    expect(generateUUID({ prefix: "conv" })).toMatch(/^conv_[0-9a-z]+$/);
    expect(generateUUID()).toMatch(/^[0-9a-z]+$/);
  });

  it("joins a suffix with the same separator", () => {
    expect(generateUUID({ prefix: "conv", suffix: "v2" })).toMatch(
      /^conv_[0-9a-z]+_v2$/,
    );
    expect(generateUUID({ prefix: "conv", separator: "-" })).toMatch(
      /^conv-[0-9a-z]+$/,
    );
  });

  it("leads with the current time in base36", () => {
    const before = Date.now();
    const body = generateUUID({ prefix: "conv" }).slice("conv_".length);
    const stamp = parseInt(body.slice(0, before.toString(36).length), 36);
    expect(stamp).toBeGreaterThanOrEqual(before - 1_000);
    expect(stamp).toBeLessThanOrEqual(Date.now() + 1_000);
  });

  it("rejects a non-positive or fractional length", () => {
    expect(() => generateUUID({ length: 0 })).toThrow(/positive integer/);
    expect(() => generateUUID({ length: 1.5 })).toThrow(/positive integer/);
  });

  it("rejects non-string prefix/suffix", () => {
    expect(() => generateUUID({ prefix: 1 as unknown as string })).toThrow(
      /must be strings/,
    );
    expect(() => generateUUID({ suffix: {} as unknown as string })).toThrow(
      /must be strings/,
    );
  });

  it("does not collide across a batch", () => {
    const ids = new Set(Array.from({ length: 2_000 }, () => genConvId()));
    expect(ids.size).toBe(2_000);
  });
});
