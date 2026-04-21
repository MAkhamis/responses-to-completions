import { randomBytes } from "node:crypto";

function rid(n = 24): string {
  return randomBytes(n).toString("base64url");
}

export const genResponseId = () => `resp_${rid()}`;
export const genConvId = () => `conv_${rid()}`;
export const genMessageId = () => `msg_${rid()}`;
export const genFcId = () => `fc_${rid()}`;
export const genCallId = () => `call_${rid(12)}`;
export const genReasoningId = () => `rs_${rid()}`;
export const genMcpListId = () => `mcpl_${rid()}`;
export const genMcpCallId = () => `mcp_${rid()}`;
export const genMcpApprovalId = () => `mcpr_${rid()}`;

export const now = () => Math.floor(Date.now() / 1000);
