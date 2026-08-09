import type { BackendAdapter } from "./backend/adapter.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatFunctionTool,
  ChatMessage,
  ChatToolCall,
  ReasoningDetail,
  UsageCostDetails,
} from "./types/completions.js";
import type {
  CreateResponseRequest,
  FunctionCallItem,
  InputItem,
  McpApprovalRequestItem,
  McpApprovalResponseItem,
  McpCallItem,
  McpListToolsItem,
  McpToolDef,
  OutputItem,
  ResponseObject,
  ToolDef,
  Usage,
} from "./types/responses.js";
import {
  McpConnection,
  needsApproval,
  type McpToolInfo,
} from "./mcp/client.js";
import {
  itemsToMessages,
  translateResponseFormat,
  translateToolChoice,
} from "./translate/request.js";
import {
  completionToOutputItems,
  translateUsage,
} from "./translate/response.js";
import { translateChunkStream, type StreamEvent } from "./translate/stream.js";
import {
  genMcpApprovalId,
  genMcpCallId,
  genMcpListId,
  now,
} from "./util/ids.js";
import type { ConversationItem } from "./store/store.js";

export interface AgentLoopDeps {
  backend: BackendAdapter;
  /** Hard cap on backend round-trips per /v1/responses call. */
  maxIterations?: number;
}

/**
 * Runs one /v1/responses request end-to-end: resolves MCP tool definitions,
 * calls the chat-completions backend, executes MCP tool calls server-side
 * (or pauses for approval), and repeats until the backend returns a response
 * with no tool calls or with a plain function_call that the client must
 * handle.
 *
 * Returns the list of OutputItems produced during this request (to be both
 * appended to conversation history and surfaced on the ResponseObject) plus
 * the aggregated usage.
 */
export class AgentLoop {
  constructor(private deps: AgentLoopDeps) {}

  /** Non-streaming execution. */
  async run(ctx: AgentRunContext): Promise<AgentRunResult> {
    if (this.deps.backend.mode === "responses") {
      return this.runViaResponses(ctx);
    }
    const complete = this.deps.backend.complete;
    if (!complete) {
      throw new Error(
        `Backend "${this.deps.backend.name}" declares mode "completions" but does not implement complete().`,
      );
    }
    const mcpSetup = await this.setupMcp(ctx.request.tools);
    const clientFunctionTools = collectClientFunctionTools(ctx.request.tools);
    const producedItems: OutputItem[] = [...mcpSetup.listItems];
    let usage: Usage | null = null;
    const maxIter = this.deps.maxIterations ?? 10;

    try {
      let messages = itemsToMessages(
        ctx.history,
        ctx.request.input,
        ctx.request.instructions,
        ctx.request.model,
      );
      let chatTools = [...clientFunctionTools, ...mcpSetup.chatTools];

      for (let iter = 0; iter < maxIter; iter++) {
        const req: ChatCompletionRequest = buildChatRequest(
          ctx.request,
          messages,
          chatTools,
        );
        const resp = await complete.call(this.deps.backend, req, ctx.signal);
        usage = mergeUsage(usage, translateUsage(resp.usage));

        const { items: newItems, outputText: _ } =
          completionToOutputItems(resp, ctx.request.model);
        producedItems.push(...newItems);

        const pendingFcs = newItems.filter(
          (it): it is FunctionCallItem => it.type === "function_call",
        );
        if (pendingFcs.length === 0) return { items: producedItems, usage };

        // Split tool calls into MCP (server-executed) vs client function tools.
        const { mcpCalls, clientCalls } = classifyCalls(
          pendingFcs,
          mcpSetup.toolToServer,
        );
        if (clientCalls.length > 0) {
          // Client must execute these; stop and return them as function_call items.
          // (Mixed batches: we still surface the MCP items we already executed.)
          return { items: producedItems, usage };
        }

        // Execute MCP calls in the order the model requested them.
        const toolMessages: ChatMessage[] = [];
        for (const fc of mcpCalls) {
          const server = mcpSetup.toolToServer.get(fc.name)!;
          const mcpOutcome = await this.invokeMcp(server, fc);
          if (mcpOutcome.kind === "approval") {
            // Emit the approval request item and stop — client must resume by
            // providing an mcp_approval_response in a follow-up request.
            producedItems.push(mcpOutcome.item);
            // Remove the surrogate function_call from produced items (we don't
            // want to expose internal plumbing when pausing for approval).
            removeFunctionCall(producedItems, fc.call_id);
            return { items: producedItems, usage };
          }
          producedItems.push(mcpOutcome.item);
          // Remove surrogate function_call: the Responses API surfaces the
          // execution as an mcp_call item, not a raw function_call.
          removeFunctionCall(producedItems, fc.call_id);
          toolMessages.push({
            role: "tool",
            tool_call_id: fc.call_id,
            content: mcpOutcome.item.output ?? mcpOutcome.item.error ?? "",
          });
        }

        // Feed results back and loop.
        messages = [
          ...messages,
          assistantToolCallMessage(
            pendingFcs,
            extractEncryptedReasoning(newItems),
          ),
          ...toolMessages,
        ];
      }

      return { items: producedItems, usage };
    } finally {
      await mcpSetup.close();
    }
  }

  /**
   * Streaming execution. Yields StreamEvents as they're produced. Multi-turn
   * agent steps are serialized: stream iteration N's deltas fully, execute
   * any MCP tool calls, emit their mcp_call items, then stream iteration N+1.
   */
  async *stream(
    ctx: AgentRunContext,
  ): AsyncGenerator<StreamEvent, AgentRunResult> {
    if (this.deps.backend.mode === "responses") {
      return yield* this.streamViaResponses(ctx);
    }
    const stream = this.deps.backend.stream;
    if (!stream) {
      throw new Error(
        `Backend "${this.deps.backend.name}" declares mode "completions" but does not implement stream().`,
      );
    }
    const mcpSetup = await this.setupMcp(ctx.request.tools);
    const clientFunctionTools = collectClientFunctionTools(ctx.request.tools);
    const producedItems: OutputItem[] = [];
    let usage: Usage | null = null;
    let seq = 0;
    const maxIter = this.deps.maxIterations ?? 10;

    // Emit mcp_list_tools items up-front so clients see which tools are available.
    for (const listItem of mcpSetup.listItems) {
      producedItems.push(listItem);
      yield {
        type: "response.output_item.added",
        sequence_number: seq++,
        output_index: producedItems.length - 1,
        item: listItem,
      };
      yield {
        type: "response.output_item.done",
        sequence_number: seq++,
        output_index: producedItems.length - 1,
        item: listItem,
      };
    }

    try {
      let messages = itemsToMessages(
        ctx.history,
        ctx.request.input,
        ctx.request.instructions,
        ctx.request.model,
      );
      let chatTools = [...clientFunctionTools, ...mcpSetup.chatTools];

      for (let iter = 0; iter < maxIter; iter++) {
        const req: ChatCompletionRequest = buildChatRequest(
          ctx.request,
          messages,
          chatTools,
        );
        const chunks = stream.call(this.deps.backend, req, ctx.signal);
        const snapshot = snapshotResponseFor(ctx, producedItems);
        const gen = translateChunkStream(
          reindexChunks(chunks, producedItems.length),
          snapshot,
          seq,
        );

        // Pipe through the translator, collecting items + usage when it finishes.
        let stepResult: { items: OutputItem[]; usage: Usage | null } = {
          items: [],
          usage: null,
        };
        while (true) {
          const r = await gen.next();
          if (r.done) {
            stepResult = r.value as {
              items: OutputItem[];
              usage: Usage | null;
            };
            break;
          }
          seq = r.value.sequence_number + 1;
          yield r.value;
        }

        usage = mergeUsage(usage, stepResult.usage);
        producedItems.push(...stepResult.items);

        const pendingFcs = stepResult.items.filter(
          (it): it is FunctionCallItem => it.type === "function_call",
        );
        if (pendingFcs.length === 0) return { items: producedItems, usage };

        const { mcpCalls, clientCalls } = classifyCalls(
          pendingFcs,
          mcpSetup.toolToServer,
        );
        if (clientCalls.length > 0) return { items: producedItems, usage };

        const toolMessages: ChatMessage[] = [];
        for (const fc of mcpCalls) {
          const server = mcpSetup.toolToServer.get(fc.name)!;
          const outcome = await this.invokeMcp(server, fc);
          removeFunctionCall(producedItems, fc.call_id);
          producedItems.push(outcome.item);
          yield {
            type: "response.output_item.added",
            sequence_number: seq++,
            output_index: producedItems.length - 1,
            item: outcome.item,
          };
          yield {
            type: "response.output_item.done",
            sequence_number: seq++,
            output_index: producedItems.length - 1,
            item: outcome.item,
          };
          if (outcome.kind === "approval") {
            return { items: producedItems, usage };
          }
          toolMessages.push({
            role: "tool",
            tool_call_id: fc.call_id,
            content: outcome.item.output ?? outcome.item.error ?? "",
          });
        }

        messages = [
          ...messages,
          assistantToolCallMessage(
            pendingFcs,
            extractEncryptedReasoning(stepResult.items),
          ),
          ...toolMessages,
        ];
      }

      return { items: producedItems, usage };
    } finally {
      await mcpSetup.close();
    }
  }

  // --- responses-endpoint pass-through ---

  /**
   * Non-streaming pass-through for adapters whose `mode === "responses"`.
   * Forwards the full request to the upstream `/responses` endpoint with the
   * locally-resolved history rolled into the `input` field, then returns the
   * upstream output as agent-result items + usage. Conversation and
   * previous_response_id are managed by the SDK's own store, so they're
   * stripped before forwarding.
   */
  private async runViaResponses(
    ctx: AgentRunContext,
  ): Promise<AgentRunResult> {
    if (!this.deps.backend.respond) {
      throw new Error(
        `Backend "${this.deps.backend.name}" declares mode "responses" but does not implement respond().`,
      );
    }
    const upstreamReq = buildResponsesPassthrough(ctx, false);
    const resp = await this.deps.backend.respond(upstreamReq, ctx.signal);
    if (resp.status === "failed" && resp.error) {
      throw new Error(`Upstream /responses failed: ${resp.error.message}`);
    }
    return { items: resp.output, usage: resp.usage };
  }

  /**
   * Streaming pass-through for adapters whose `mode === "responses"`.
   * Forwards upstream `StreamEvent`s out unchanged with one exception: we
   * swallow the upstream-emitted lifecycle events (`response.created`,
   * `response.in_progress`, `response.completed`, `response.failed`) because
   * the `StreamResponse` wrapper in the SDK client emits its own with the
   * SDK-side id and conversation. Items + usage are accumulated from
   * `response.output_item.done` and `response.completed`.
   */
  private async *streamViaResponses(
    ctx: AgentRunContext,
  ): AsyncGenerator<StreamEvent, AgentRunResult> {
    if (!this.deps.backend.respondStream) {
      throw new Error(
        `Backend "${this.deps.backend.name}" declares mode "responses" but does not implement respondStream().`,
      );
    }
    const upstreamReq = buildResponsesPassthrough(ctx, true);
    const items: OutputItem[] = [];
    let usage: Usage | null = null;
    let seq = 0;

    for await (const ev of this.deps.backend.respondStream(
      upstreamReq,
      ctx.signal,
    )) {
      if (ev.type === "response.completed") {
        if (ev.response.output?.length) {
          items.splice(0, items.length, ...ev.response.output);
        }
        if (ev.response.usage) usage = ev.response.usage;
        continue;
      }
      if (ev.type === "response.failed") {
        throw new Error(
          `Upstream /responses failed: ${ev.response.error?.message ?? "unknown"}`,
        );
      }
      if (
        ev.type === "response.created" ||
        ev.type === "response.in_progress"
      ) {
        continue;
      }
      if (ev.type === "response.output_item.done") {
        items.push(ev.item);
      }
      yield { ...ev, sequence_number: seq++ };
    }
    return { items, usage };
  }

  // --- helpers ---

  private async setupMcp(tools: ToolDef[] | undefined): Promise<{
    listItems: McpListToolsItem[];
    chatTools: ChatFunctionTool[];
    toolToServer: Map<string, ResolvedMcpServer>;
    close: () => Promise<void>;
  }> {
    const listItems: McpListToolsItem[] = [];
    const chatTools: ChatFunctionTool[] = [];
    const toolToServer = new Map<string, ResolvedMcpServer>();
    const connections: McpConnection[] = [];

    for (const tool of tools ?? []) {
      if (tool.type !== "mcp") continue;
      const conn = new McpConnection(tool);
      connections.push(conn);
      let toolInfos: McpToolInfo[];
      try {
        toolInfos = await conn.listTools();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        listItems.push({
          type: "mcp_list_tools",
          id: genMcpListId(),
          server_label: tool.server_label,
          tools: [],
          // OpenAI's shape doesn't carry an error field here; we surface via metadata
          // conventions most clients ignore, but callers can inspect it.
          ...({ error: msg } as Record<string, unknown>),
        } as McpListToolsItem);
        continue;
      }

      listItems.push({
        type: "mcp_list_tools",
        id: genMcpListId(),
        server_label: tool.server_label,
        tools: toolInfos,
      });

      const server: ResolvedMcpServer = {
        def: tool,
        connection: conn,
        toolNames: new Set(toolInfos.map((t) => t.name)),
      };
      for (const info of toolInfos) {
        toolToServer.set(info.name, server);
        chatTools.push({
          type: "function",
          function: {
            name: info.name,
            description: info.description,
            parameters: info.input_schema,
          },
        });
      }
    }

    return {
      listItems,
      chatTools,
      toolToServer,
      close: async () => {
        await Promise.all(connections.map((c) => c.close()));
      },
    };
  }

  private async invokeMcp(
    server: ResolvedMcpServer,
    fc: FunctionCallItem,
  ): Promise<
    | { kind: "done"; item: McpCallItem }
    | { kind: "approval"; item: McpApprovalRequestItem }
  > {
    if (needsApproval(server.def.require_approval, fc.name)) {
      const approval: McpApprovalRequestItem = {
        type: "mcp_approval_request",
        id: genMcpApprovalId(),
        server_label: server.def.server_label,
        name: fc.name,
        arguments: fc.arguments,
      };
      return { kind: "approval", item: approval };
    }

    let parsedArgs: Record<string, unknown> = {};
    try {
      parsedArgs = fc.arguments
        ? (JSON.parse(fc.arguments) as Record<string, unknown>)
        : {};
    } catch {
      parsedArgs = {};
    }

    try {
      const result = await server.connection.call(fc.name, parsedArgs);
      const item: McpCallItem = {
        type: "mcp_call",
        id: genMcpCallId(),
        server_label: server.def.server_label,
        name: fc.name,
        arguments: fc.arguments,
        output: result.isError ? null : result.output,
        error: result.isError ? result.output : null,
        approval_request_id: null,
      };
      return { kind: "done", item };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        kind: "done",
        item: {
          type: "mcp_call",
          id: genMcpCallId(),
          server_label: server.def.server_label,
          name: fc.name,
          arguments: fc.arguments,
          output: null,
          error: msg,
          approval_request_id: null,
        },
      };
    }
  }
}

// ---- context / result types ----

export interface AgentRunContext {
  request: CreateResponseRequest;
  history: ConversationItem[];
  signal?: AbortSignal;
}

export interface AgentRunResult {
  items: OutputItem[];
  usage: Usage | null;
}

interface ResolvedMcpServer {
  def: McpToolDef;
  connection: McpConnection;
  toolNames: Set<string>;
}

// ---- helpers ----

function collectClientFunctionTools(
  tools: ToolDef[] | undefined,
): ChatFunctionTool[] {
  return (tools ?? [])
    .filter(
      (t): t is Extract<ToolDef, { type: "function" }> => t.type === "function",
    )
    .map((t) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
        ...(t.strict !== undefined ? { strict: t.strict } : {}),
      },
    }));
}

function buildChatRequest(
  r: CreateResponseRequest,
  messages: ChatMessage[],
  tools: ChatFunctionTool[],
): ChatCompletionRequest {
  return {
    model: r.model,
    messages,
    ...(tools.length ? { tools } : {}),
    ...(r.tool_choice !== undefined
      ? { tool_choice: translateToolChoice(r.tool_choice) }
      : {}),
    ...(tools.length && r.parallel_tool_calls !== undefined
      ? { parallel_tool_calls: r.parallel_tool_calls }
      : {}),
    ...(r.temperature !== undefined ? { temperature: r.temperature } : {}),
    ...(r.top_p !== undefined ? { top_p: r.top_p } : {}),
    ...(r.max_output_tokens !== undefined
      ? { max_tokens: r.max_output_tokens }
      : {}),
    ...(r.user !== undefined ? { user: r.user } : {}),
    ...(r.metadata ? { metadata: r.metadata } : {}),
    ...(() => {
      const rf = translateResponseFormat(r.text);
      return rf ? { response_format: rf } : {};
    })(),
    ...(r.reasoning?.effort ? { reasoning_effort: r.reasoning.effort } : {}),
    ...(r.service_tier !== undefined ? { service_tier: r.service_tier } : {}),
  };
}

function classifyCalls(
  fcs: FunctionCallItem[],
  mcp: Map<string, ResolvedMcpServer>,
): { mcpCalls: FunctionCallItem[]; clientCalls: FunctionCallItem[] } {
  const mcpCalls: FunctionCallItem[] = [];
  const clientCalls: FunctionCallItem[] = [];
  for (const fc of fcs) {
    if (mcp.has(fc.name)) mcpCalls.push(fc);
    else clientCalls.push(fc);
  }
  return { mcpCalls, clientCalls };
}

function assistantToolCallMessage(
  fcs: FunctionCallItem[],
  reasoning?: ReasoningDetail[] | null,
): ChatMessage {
  const tool_calls: ChatToolCall[] = fcs.map((fc) => ({
    id: fc.call_id,
    type: "function",
    function: { name: fc.name, arguments: fc.arguments ?? "" },
  }));
  const msg: ChatMessage = { role: "assistant", content: null, tool_calls };
  if (reasoning?.length)
    (msg as { reasoning_details?: ReasoningDetail[] }).reasoning_details =
      reasoning;
  return msg;
}

function extractEncryptedReasoning(
  items: OutputItem[],
): ReasoningDetail[] | null {
  const all: ReasoningDetail[] = [];
  for (const item of items) {
    if (item.type !== "reasoning") continue;
    const r = item as import("./types/responses.js").ReasoningItem;
    if (!r.encrypted_content) continue;
    try {
      const parsed = JSON.parse(r.encrypted_content);
      if (Array.isArray(parsed)) all.push(...(parsed as ReasoningDetail[]));
    } catch {}
  }
  return all.length > 0 ? all : null;
}

function removeFunctionCall(items: OutputItem[], callId: string): void {
  const idx = items.findIndex(
    (it) =>
      it.type === "function_call" &&
      (it as FunctionCallItem).call_id === callId,
  );
  if (idx >= 0) items.splice(idx, 1);
}

function mergeUsage(a: Usage | null, b: Usage | null): Usage | null {
  if (!a) return b;
  if (!b) return a;
  return {
    input_tokens: a.input_tokens + b.input_tokens,
    output_tokens: a.output_tokens + b.output_tokens,
    total_tokens: a.total_tokens + b.total_tokens,
    ...(a.cost !== undefined || b.cost !== undefined
      ? { cost: (a.cost ?? 0) + (b.cost ?? 0) }
      : {}),
    ...(a.cost_details || b.cost_details
      ? { cost_details: mergeCostDetails(a.cost_details, b.cost_details) }
      : {}),
  };
}

function mergeCostDetails(
  a?: UsageCostDetails,
  b?: UsageCostDetails,
): UsageCostDetails {
  const sum = (x?: number | null, y?: number | null): number | undefined =>
    x != null || y != null ? (x ?? 0) + (y ?? 0) : undefined;
  const upstream = sum(a?.upstream_inference_cost, b?.upstream_inference_cost);
  const prompt = sum(
    a?.upstream_inference_prompt_cost,
    b?.upstream_inference_prompt_cost,
  );
  const completions = sum(
    a?.upstream_inference_completions_cost,
    b?.upstream_inference_completions_cost,
  );
  return {
    ...(upstream !== undefined ? { upstream_inference_cost: upstream } : {}),
    ...(prompt !== undefined
      ? { upstream_inference_prompt_cost: prompt }
      : {}),
    ...(completions !== undefined
      ? { upstream_inference_completions_cost: completions }
      : {}),
  };
}

function snapshotResponseFor(
  ctx: AgentRunContext,
  produced: OutputItem[],
): ResponseObject {
  const r = ctx.request;
  return {
    id: "pending",
    object: "response",
    created_at: now(),
    status: "in_progress",
    error: null,
    incomplete_details: null,
    instructions: r.instructions ?? null,
    max_output_tokens: r.max_output_tokens ?? null,
    model: r.model,
    output: produced,
    parallel_tool_calls: r.parallel_tool_calls ?? true,
    previous_response_id: r.previous_response_id ?? null,
    conversation:
      typeof r.conversation === "string"
        ? { id: r.conversation }
        : (r.conversation ?? null),
    service_tier: r.service_tier ?? null,
    temperature: r.temperature ?? null,
    tool_choice: r.tool_choice ?? "auto",
    tools: r.tools ?? [],
    top_p: r.top_p ?? null,
    usage: null,
    user: r.user ?? null,
    metadata: r.metadata ?? null,
  };
}

function buildResponsesPassthrough(
  ctx: AgentRunContext,
  stream: boolean,
): CreateResponseRequest {
  const inputItems = combineHistoryAndInput(ctx.history, ctx.request.input);
  const {
    conversation: _c,
    previous_response_id: _p,
    store: _s,
    stream: _st,
    signal: _sig,
    ...rest
  } = ctx.request as CreateResponseRequest & { signal?: AbortSignal };
  return { ...rest, input: inputItems, stream };
}

function combineHistoryAndInput(
  history: ConversationItem[],
  input: string | InputItem[] | undefined,
): InputItem[] {
  const out: InputItem[] = [...(history as InputItem[])];
  if (input !== undefined) {
    if (typeof input === "string") {
      out.push({ type: "message", role: "user", content: input });
    } else {
      out.push(...input);
    }
  }
  return out;
}

/**
 * Later iterations of the agent loop start numbering their chunks from 0
 * again (since each backend call is a fresh stream), but Responses-API
 * output_index is a monotonic counter across the whole response. The stream
 * translator uses its own local nextOutputIndex; we feed it a positional
 * offset implicitly by treating each iteration as a fresh translator instance
 * and then the caller re-indexes produced items when concatenating. For the
 * event sequence_number we use a single monotonic counter in the outer loop.
 */
async function* reindexChunks<T>(
  src: AsyncIterable<T>,
  _baseIndex: number,
): AsyncIterable<T> {
  for await (const x of src) yield x;
}
