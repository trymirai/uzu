import type { ParsedPatch, SessionOutputStats } from "@/types/llm-stream";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import { assistantHistoryMessage, joinTurns, parsedFrom, sanitize, type HistoryMessage } from "./text";
import { messageReasoning, messageText, Transcript, type Reply, type ReplyStats } from "./transcript";

export function mapReplyStats(stats: ReplyStats | undefined): SessionOutputStats {
  const prefillDuration = stats?.timeToFirstToken ?? 0;
  const totalDuration = stats?.duration ?? 0;
  const inputTokens = stats?.tokensCountInput ?? 0;
  const outputTokens = stats?.tokensCountOutput ?? 0;
  return {
    prefillStats: {
      duration: prefillDuration,
      tokensCount: inputTokens,
      tokensPerSecond: stats?.prefillTokensPerSecond ?? 0,
    },
    generateStats: {
      duration: Math.max(0, totalDuration - prefillDuration),
      tokensCount: outputTokens,
      tokensPerSecond: stats?.generateTokensPerSecond ?? 0,
    },
    totalStats: { duration: totalDuration, tokensCountInput: inputTokens, tokensCountOutput: outputTokens },
    ...(stats?.tokensPerForwardPass !== undefined ? { tokensPerForwardPass: stats.tokensPerForwardPass } : {}),
  };
}

/** The visible text of a run, which can span several assistant turns around tool calls. */
export class ReplyText {
  private completedText = "";
  private completedReasoning = "";
  text = "";
  reasoning = "";
  private toolTurnFinished = false;

  update(reply: Reply) {
    // Each automatic tool continuation starts a new snapshot at zero.
    // Preserve visible content from the completed assistant turn first.
    if (this.toolTurnFinished) {
      this.completedText = joinTurns(this.completedText, this.text);
      this.completedReasoning = joinTurns(this.completedReasoning, this.reasoning);
    }
    this.text = messageText(reply.message);
    this.reasoning = messageReasoning(reply.message);
    this.toolTurnFinished = reply.finishReason === "ToolCalls";
  }

  raw(): [text: string, reasoning: string] {
    return [joinTurns(this.completedText, this.text), joinTurns(this.completedReasoning, this.reasoning)];
  }

  effective(promoteReasoning: boolean): [text: string, reasoning: string] {
    // Tool-call reasoning is internal even when a model ignored the
    // disabled-reasoning preference for its final answer.
    return promoteReasoning ? [joinTurns(this.completedText, this.reasoning), this.completedReasoning] : this.raw();
  }
}

function finalParsed(text: string, reasoning: string, promotedReasoning: boolean): ParsedPatch | undefined {
  const parsed = parsedFrom(text, reasoning);
  // The frontend merges parsed patches. Explicitly clear reasoning that was
  // streamed before we knew it belonged in the final answer.
  return parsed && promotedReasoning ? { ...parsed, chainOfThought: sanitize(reasoning.trim()) } : parsed;
}

/** Turns uzu's stream chunks into client run events, like desktop/src/chat/stream.rs. */
export class ReplyRun {
  private output?: Reply;
  private readonly replyText = new ReplyText();
  private readonly transcript = new Transcript();

  constructor(
    private readonly emit: (event: RunEvent) => void,
    // Disabled reasoning: a model that still reasons gets its reasoning shown as the answer.
    private readonly promoteReasoning: boolean,
    private readonly toolsEnabled: boolean,
  ) {}

  reply(reply: Reply) {
    const hasText = reply.message.content.some((block) => block.type === "text" && block.value !== "");
    const hasReasoning = reply.message.content.some((block) => block.type === "reasoning" && block.value !== "");
    const promoteWhileStreaming = !this.toolsEnabled && this.promoteReasoning && !hasText && hasReasoning;
    const event = this.transcript.streamUpdate(this.output, reply, promoteWhileStreaming);
    if (event) this.emit(event);
    if (reply.finishReason) this.replyText.update(reply);
    this.output = reply;
  }

  toolResults(messages: Reply["message"][]) {
    this.transcript.toolResults(messages);
    this.emit({ type: "transcript", items: this.transcript.items(false) });
  }

  /** Publishes the transcript as it stood when the stream failed. */
  fail() {
    this.transcript.flushReply(this.output);
    this.emit({ type: "transcript", items: this.transcript.items(true) });
  }

  /**
   * Emits the final event. Returns the assistant message that the resident
   * session now holds, when the client will persist exactly that reply.
   */
  finish(cancelled: boolean, chatName: string | undefined): HistoryMessage | undefined {
    this.transcript.flushReply(this.output);
    if (this.output && !this.output.finishReason) this.replyText.update(this.output);
    const finishReason = this.output?.finishReason;
    const [rawText, rawReasoning] = this.replyText.raw();
    const promoted =
      this.promoteReasoning &&
      this.replyText.text === "" &&
      this.replyText.reasoning !== "" &&
      finishReason !== undefined &&
      finishReason !== "ToolCalls";
    const [text, reasoning] = this.replyText.effective(promoted);
    this.transcript.setReasoningPromotion(promoted);
    this.emit({
      type: "done",
      text: sanitize(text),
      stats: mapReplyStats(this.output?.stats),
      // Cancellation between a tool result and the next token has no assistant
      // snapshot of its own; the last reply still says ToolCalls.
      finishReason: cancelled ? "Cancelled" : finishReason,
      parsed: finalParsed(text, reasoning, promoted),
      chatName,
      transcript: this.transcript.items(true),
    });
    // Cancelled or truncated replies may diverge from what the client persists.
    return !cancelled && finishReason === "Stop" ? assistantHistoryMessage(rawText, rawReasoning, promoted) : undefined;
  }
}
