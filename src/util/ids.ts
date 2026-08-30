import { randomBytes } from "node:crypto";

function rid(n = 24): string {
  return randomBytes(n).toString("base64url");
}

export interface GenerateUUIDOptions {
  /** Leading label, joined with `separator` (e.g. `"conv"` → `conv_…`). */
  prefix?: string;
  /** Trailing label, joined with `separator`. */
  suffix?: string;
  /** Bytes of randomness in the tail. */
  length?: number;
  /** Character joining prefix/suffix to the body. */
  separator?: string;
}

export function generateUUID({
  prefix,
  suffix,
  length = 8,
  separator = "_",
}: GenerateUUIDOptions = {}): string {
  if (
    (prefix && typeof prefix !== "string") ||
    (suffix && typeof suffix !== "string")
  ) {
    throw new Error("Prefix and suffix must be strings");
  }
  if (!Number.isInteger(length) || length < 1) {
    throw new Error("length must be a positive integer");
  }

  const randomPortion = BigInt(
    `0x${randomBytes(length).toString("hex")}`,
  ).toString(36);
  const timestamp = Date.now().toString(36);

  const formattedPrefix = prefix ? `${prefix}${separator}` : "";
  const formattedSuffix = suffix ? `${separator}${suffix}` : "";

  return `${formattedPrefix}${timestamp}${randomPortion}${formattedSuffix}`;
}

export const genResponseId = () => `resp_${rid()}`;
export const genConvId = () => generateUUID({ prefix: "conv" });
export const genMessageId = () => `msg_${rid()}`;
export const genFcId = () => `fc_${rid()}`;
export const genCallId = () => `call_${rid(12)}`;
export const genReasoningId = () => `rs_${rid()}`;
export const genMcpListId = () => `mcpl_${rid()}`;
export const genMcpCallId = () => `mcp_${rid()}`;
export const genMcpApprovalId = () => `mcpr_${rid()}`;

export const now = () => Math.floor(Date.now() / 1000);
