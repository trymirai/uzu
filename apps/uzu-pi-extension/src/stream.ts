import {
  createAssistantMessageEventStream,
  getCurrentTools,
  parseStreamingJson,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import {
  ChatContentBlockToolCallCandidate,
  ChatContentBlockToolCall,
  ChatReplyConfig,
  ChatReplyFinishReason,
  ChatSessionStreamChunkError,
  ChatSessionStreamChunkReplies,
  SamplingMethodGreedy,
  SamplingMethodStochastic,
  ToolCallArgumentStream,
  type ToolNamespace,
  type ChatModelInfo,
  type ChatReply,
  type ChatSessionStream,
  type Model as UzuModel,
} from "@trymirai/uzu";
import { SessionCache } from "./cache.js";
import { nativeTools, prepareInput, type Input } from "./messages.js";
import type { api } from "./models.js";

export function emptyMessage(model: Model<string>): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    stopReason: "pending",
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  };
}

export class ReplyEvents {
  private thinkingIndex?: number;
  private textIndex?: number;
  private calls = new Map<
    string,
    {
      index?: number;
      stream: ToolCallArgumentStream;
      json: string;
      complete: boolean;
      source?: string;
    }
  >();

  constructor(
    readonly message: AssistantMessage,
    private readonly stream: AssistantMessageEventStream,
    private readonly tools: ToolNamespace[],
  ) {}

  update(reply: ChatReply) {
    if (reply.finishReason === ChatReplyFinishReason.ContextLimitReached)
      throw new Error("context_length_exceeded: Uzu context limit reached");
    this.text("thinking", reply.message.reasoning ?? "");
    this.text("text", reply.message.text ?? "");
    for (const content of reply.message.content) {
      if (
        !(content instanceof ChatContentBlockToolCall) &&
        !(content instanceof ChatContentBlockToolCallCandidate)
      )
        continue;
      const id =
        content instanceof ChatContentBlockToolCall
          ? content.value.identifier
          : content.identifier;
      if (!id) throw new Error("Uzu returned a tool call without an ID");
      let call = this.calls.get(id);
      if (!call) {
        call = {
          stream: ToolCallArgumentStream.create(this.tools),
          json: "",
          complete: false,
        };
        this.calls.set(id, call);
      }
      if (call.complete) {
        if (
          !(content instanceof ChatContentBlockToolCall) ||
          call.source !==
            JSON.stringify([content.value.name, content.value.arguments.json])
        )
          throw new Error("Uzu revised a completed tool call");
        continue;
      }
      let delta;
      if (content instanceof ChatContentBlockToolCallCandidate) {
        const value: unknown = JSON.parse(content.value.json);
        delta = call.stream.update(
          typeof value === "string" ? value : content.value.json,
        );
      } else {
        delta = call.stream.finish(content.value);
        call.complete = true;
        call.source = JSON.stringify([
          content.value.name,
          content.value.arguments.json,
        ]);
      }
      if (!delta.name) continue;
      if (call.index === undefined) {
        call.index = this.message.content.length;
        this.message.content.push({
          type: "toolCall",
          id,
          name: delta.name,
          arguments: {},
        });
        this.stream.push({
          type: "toolcall_start",
          contentIndex: call.index,
          partial: this.message,
        });
      }
      const block = this.message.content[call.index];
      if (block?.type !== "toolCall")
        throw new Error("Invalid tool content index");
      call.json += delta.arguments;
      block.arguments = call.complete
        ? JSON.parse(call.json)
        : parseStreamingJson(call.json);
      if (!delta.arguments) continue;
      this.stream.push({
        type: "toolcall_delta",
        contentIndex: call.index,
        delta: delta.arguments,
        partial: this.message,
      });
    }
    const cached = reply.stats.tokensCountInputCached ?? 0;
    const input = reply.stats.tokensCountInput ?? 0;
    this.message.usage.input = input;
    this.message.usage.cacheRead = cached;
    this.message.usage.output = reply.stats.tokensCountOutput ?? 0;
    this.message.usage.totalTokens =
      this.message.usage.input + cached + this.message.usage.output;
  }

  private text(kind: "text" | "thinking", value: string) {
    let index = kind === "text" ? this.textIndex : this.thinkingIndex;
    if (index === undefined) {
      if (!value) return;
      index = this.message.content.length;
      if (kind === "text") {
        this.textIndex = index;
        this.message.content.push({ type: "text", text: "" });
        this.stream.push({
          type: "text_start",
          contentIndex: index,
          partial: this.message,
        });
      } else {
        this.thinkingIndex = index;
        this.message.content.push({ type: "thinking", thinking: "" });
        this.stream.push({
          type: "thinking_start",
          contentIndex: index,
          partial: this.message,
        });
      }
    }
    const block = this.message.content[index]!;
    if (block.type === "toolCall") throw new Error("Invalid content index");
    const previous = block.type === "text" ? block.text : block.thinking;
    if (!value.startsWith(previous))
      throw new Error(`Uzu revised streamed ${kind}`);
    const delta = value.slice(previous.length);
    if (!delta) return;
    if (block.type === "text") {
      block.text = value;
      this.stream.push({
        type: "text_delta",
        contentIndex: index,
        delta,
        partial: this.message,
      });
    } else {
      block.thinking = value;
      this.stream.push({
        type: "thinking_delta",
        contentIndex: index,
        delta,
        partial: this.message,
      });
    }
  }

  finish(reply: ChatReply) {
    switch (reply.finishReason) {
      case ChatReplyFinishReason.Stop:
        this.message.stopReason = "stop";
        break;
      case ChatReplyFinishReason.Length:
        this.message.stopReason = "length";
        break;
      case ChatReplyFinishReason.ToolCalls:
        if (!this.calls.size)
          throw new Error("Uzu returned no complete tool calls");
        this.message.stopReason = "toolUse";
        break;
      case ChatReplyFinishReason.ContextLimitReached:
        throw new Error("context_length_exceeded: Uzu context limit reached");
      case ChatReplyFinishReason.Cancelled:
        throw new DOMException("Uzu generation cancelled", "AbortError");
      case ChatReplyFinishReason.Rejected:
        throw new Error("Uzu rejected the request");
      default:
        throw new Error("Uzu stream ended without a finish reason");
    }
    if (
      reply.message.content.some(
        (block) => block instanceof ChatContentBlockToolCallCandidate,
      )
    ) {
      throw new Error("Uzu returned an unfinished tool call");
    }
    if (reply.message.toolCalls.length !== this.calls.size)
      throw new Error("Uzu removed a streamed tool call");
    if (
      this.calls.size &&
      reply.finishReason !== ChatReplyFinishReason.ToolCalls
    )
      throw new Error("Uzu stopped before completing its tool calls");
    this.message.content.forEach((block, index) => {
      if (block.type === "text")
        this.stream.push({
          type: "text_end",
          contentIndex: index,
          content: block.text,
          partial: this.message,
        });
      else if (block.type === "thinking")
        this.stream.push({
          type: "thinking_end",
          contentIndex: index,
          content: block.thinking,
          partial: this.message,
        });
      else {
        this.stream.push({
          type: "toolcall_end",
          contentIndex: index,
          toolCall: block,
          partial: this.message,
        });
      }
    });
  }
}

export interface ModelEntry {
  model: UzuModel;
  info: ChatModelInfo;
  definition: Model<typeof api>;
}

export function createStream(
  cache: SessionCache,
  resolveEntry: (
    id: string,
    signal?: AbortSignal,
  ) => Promise<ModelEntry | undefined>,
) {
  return (
    model: Model<string>,
    context: TranscriptContext,
    options: SimpleStreamOptions = {},
  ): AssistantMessageEventStream => {
    const stream = createAssistantMessageEventStream();
    const message = emptyMessage(model);
    void cache
      .exclusive(options.signal, async () => {
        let native: ChatSessionStream | undefined;
        const abort = () => native?.cancelToken.cancel();
        options.signal?.addEventListener("abort", abort, { once: true });
        try {
          if (options.deferred)
            throw new Error("Uzu does not support deferred generation");
          if (options.fetch)
            throw new Error(
              "Uzu runs locally and cannot use a custom fetch implementation",
            );
          if (
            options.toolChoice !== undefined &&
            options.toolChoice !== "auto" &&
            options.toolChoice !== "none"
          )
            throw new Error('Uzu supports only toolChoice "auto" and "none"');
          if (options.toolChoice !== "none") {
            for (const tool of getCurrentTools(context.messages)) {
              if (
                tool.constrainedSampling &&
                tool.constrainedSampling.type === "json_schema" &&
                tool.constrainedSampling.strict === "require"
              )
                throw new Error(
                  `Tool "${tool.name}" requires JSON-schema constrained sampling, which this Uzu adapter does not support`,
                );
            }
          }
          const tokenLimit = options.maxTokens;
          if (
            tokenLimit !== undefined &&
            (!Number.isSafeInteger(tokenLimit) ||
              tokenLimit < 1 ||
              tokenLimit > 0xffffffff)
          )
            throw new Error(
              "maxTokens must be an integer between 1 and 4294967295",
            );
          const entry = await resolveEntry(model.id, options.signal);
          if (!entry) throw new Error(`Uzu model is unavailable: ${model.id}`);
          let input = prepareInput(context, entry.info, options);
          let config = ChatReplyConfig.create()
            .withToolTurnLimit(0)
            .withTokenLimit(tokenLimit);
          if (options.temperature !== undefined) {
            if (
              !Number.isFinite(options.temperature) ||
              options.temperature < 0
            )
              throw new Error("Invalid temperature");
            config = config.withSamplingMethod(
              options.temperature === 0
                ? new SamplingMethodGreedy()
                : new SamplingMethodStochastic(options.temperature),
            );
          }
          const replacement = await options.onPayload?.(
            { input, config },
            model,
          );
          if (replacement !== undefined) {
            if (
              !replacement ||
              typeof replacement !== "object" ||
              !("input" in replacement) ||
              !("config" in replacement) ||
              !(replacement.config instanceof ChatReplyConfig)
            )
              throw new Error(
                "onPayload must return { input, config: ChatReplyConfig }",
              );
            input = replacement.input as Input;
            config = replacement.config;
          }
          options.signal?.throwIfAborted();
          const events = new ReplyEvents(message, stream, nativeTools(input));
          const prepared = await cache.prepare(
            entry.model,
            options.sessionId,
            input,
            options.cacheRetention !== "none",
          );
          options.signal?.throwIfAborted();
          native = await prepared.session.replyWithStream(
            prepared.messages,
            config,
          );
          if (options.signal?.aborted) abort();
          stream.push({ type: "start", partial: message });
          let last: ChatReply | undefined;
          for await (const chunk of native) {
            await options.onProviderStreamEvent?.(chunk, model);
            options.signal?.throwIfAborted();
            if (chunk instanceof ChatSessionStreamChunkError) {
              if (
                chunk.error.type === "Backend" &&
                chunk.error.message === "Context overflow"
              )
                throw new Error(
                  "context_length_exceeded: Uzu input exceeds the context window",
                );
              throw new Error(JSON.stringify(chunk.error));
            }
            if (chunk instanceof ChatSessionStreamChunkReplies) {
              if (chunk.replies.length !== 1)
                throw new Error(
                  "Expected one Uzu reply with automatic tools disabled",
                );
              last = chunk.replies[0]!;
              events.update(last);
            } else
              throw new Error("Unexpected automatic tool execution in Uzu");
          }
          options.signal?.throwIfAborted();
          if (!last) throw new Error("Uzu returned no response");
          if (
            last.finishReason === ChatReplyFinishReason.ToolCalls &&
            !entry.info.capabilities.supportsMultipleToolCalls &&
            last.message.toolCalls.length > 1
          )
            throw new Error(
              "This Uzu model returned more tool calls than its encoding supports",
            );
          events.finish(last);
          cache.complete(message);
          if (
            options.cacheRetention === "none" ||
            options.sessionId === undefined
          )
            cache.invalidate();
          stream.push({
            type: "done",
            reason: message.stopReason as "stop" | "length" | "toolUse",
            message,
          });
        } catch (error) {
          abort();
          // Drain cancellation before releasing the lock or model. Native tasks own
          // their state until the stream closes, even after an observer throws.
          if (native) {
            try {
              while ((await native.next()) !== null) {}
            } catch {}
          }
          cache.invalidate();
          throw error;
        } finally {
          options.signal?.removeEventListener("abort", abort);
        }
      })
      .catch((error: unknown) => {
        message.stopReason =
          options.signal?.aborted ||
          (error instanceof Error && error.name === "AbortError")
            ? "aborted"
            : "error";
        message.errorMessage =
          error instanceof Error ? error.message : String(error);
        stream.push({
          type: "error",
          reason: message.stopReason,
          error: message,
        });
      })
      .finally(() => stream.end());
    return stream;
  };
}
