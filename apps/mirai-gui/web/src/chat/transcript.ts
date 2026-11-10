import { isChartSpec } from "@/types/chart";
import type { SessionOutputFinishReason, TranscriptItem } from "@/types/llm-stream";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import { sanitize } from "./text";

// Plain copies of uzu's chat values, so the stream logic does not hold wasm objects.
export type ToolCallValue = { identifier?: string; name: string; arguments: string };
export type ContentBlock =
  | { type: "text"; value: string }
  | { type: "reasoning"; value: string }
  | { type: "toolCall"; call: ToolCallValue }
  | { type: "toolResult"; identifier?: string; name?: string; value: string }
  | { type: "other" };
export type TranscriptMessage = { role: "assistant" | "tool" | "other"; metadata: string; content: ContentBlock[] };
export type ReplyStats = {
  duration: number;
  timeToFirstToken?: number;
  prefillTokensPerSecond?: number;
  generateTokensPerSecond?: number;
  tokensCountInput?: number;
  tokensCountOutput?: number;
  tokensPerForwardPass?: number;
};
export type Reply = { message: TranscriptMessage; finishReason?: SessionOutputFinishReason; stats?: ReplyStats };

const joinBlocks = (message: TranscriptMessage, type: "text" | "reasoning"): string =>
  message.content.map((block) => (block.type === type ? block.value : "")).join("");

export const messageText = (message: TranscriptMessage): string => joinBlocks(message, "text");
export const messageReasoning = (message: TranscriptMessage): string => joinBlocks(message, "reasoning");

// Match ToolDescriptor's boolean coercion for markup-based templates.
export function hiddenCall(call: ToolCallValue): boolean {
  if (call.name !== "set_chat_name") return false;
  try {
    const hidden: unknown = (JSON.parse(call.arguments) as { hidden?: unknown } | null)?.hidden;
    return hidden === true || (typeof hidden === "string" && hidden.trim().toLowerCase() === "true");
  } catch {
    return false;
  }
}

const isVisible = (block: ContentBlock): boolean =>
  block.type === "text" || block.type === "reasoning"
    ? block.value.trim() !== ""
    : block.type === "toolCall" && !hiddenCall(block.call);

const sameBlock = (a: ContentBlock, b: ContentBlock): boolean => JSON.stringify(a) === JSON.stringify(b);

/**
 * Return only an appended suffix, or require a snapshot if any visible block
 * changes shape. Invisible candidates and housekeeping calls do not interrupt
 * streaming. A decoder correction (including a replaced partial Unicode
 * character) falls back to a snapshot rather than using offsets blindly.
 */
export function visibleDelta(
  previous: TranscriptMessage,
  current: TranscriptMessage,
  promoteReasoning: boolean,
): string | undefined {
  if (previous.role !== "assistant" || current.role !== previous.role || current.metadata !== previous.metadata)
    return undefined;
  const before = previous.content.filter(isVisible);
  const after = current.content.filter(isVisible);
  if (before.length !== after.length) return undefined;
  for (const [index, old] of before.entries()) {
    const next = after[index]!;
    if (sameBlock(old, next)) continue;
    if (index !== before.length - 1) return undefined;
    let from: string;
    let to: string;
    if (old.type === "text" && next.type === "text") [from, to] = [old.value, next.value];
    else if (old.type === "reasoning" && next.type === "reasoning")
      [from, to] = promoteReasoning ? [old.value, next.value] : [old.value.trim(), next.value.trim()];
    else return undefined;
    return to.startsWith(from) ? sanitize(to.slice(from.length)) : undefined;
  }
  return "";
}

function completeThinking(items: TranscriptItem[]) {
  const last = items.at(-1);
  if (last?.type === "thinking") last.completed = true;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

/**
 * Only this run's messages: replayed history belongs to earlier UI messages.
 * Result events arrive before the next prefill, so completion never waits for
 * another generated token.
 */
export class Transcript {
  private messages: TranscriptMessage[] = [];
  private finishReason?: SessionOutputFinishReason;
  private promoteReasoning = false;
  private streamSnapshotValid = false;
  private lastItemIndex?: number;
  private replyDirty = false;

  /**
   * Append to the last visible item while the current message keeps its
   * structure. The caller owns the latest reply; materialize it only when a
   * snapshot or final/error transcript needs the complete message history.
   */
  streamUpdate(previous: Reply | undefined, reply: Reply, promoteReasoning: boolean): RunEvent | undefined {
    if (
      this.streamSnapshotValid &&
      this.promoteReasoning === promoteReasoning &&
      !reply.finishReason &&
      previous &&
      !previous.finishReason
    ) {
      const delta = visibleDelta(previous.message, reply.message, promoteReasoning);
      if (delta !== undefined && (delta === "" || this.lastItemIndex !== undefined)) {
        this.replyDirty = true;
        return delta ? { type: "transcriptDelta", index: this.lastItemIndex!, delta } : undefined;
      }
    }
    this.update(reply);
    this.promoteReasoning = promoteReasoning;
    const items = this.items(false);
    this.lastItemIndex = items.length > 0 ? items.length - 1 : undefined;
    this.streamSnapshotValid = true;
    return { type: "transcript", items };
  }

  flushReply(reply: Reply | undefined) {
    if (this.replyDirty && reply) this.update(reply);
  }

  update(reply: Reply) {
    if (this.finishReason || this.messages.length === 0) this.messages.push(reply.message);
    else this.messages[this.messages.length - 1] = reply.message;
    this.finishReason = reply.finishReason;
    this.replyDirty = false;
    this.streamSnapshotValid = false;
  }

  toolResults(messages: TranscriptMessage[]) {
    this.messages.push(...messages);
    this.finishReason = "ToolCalls";
    this.streamSnapshotValid = false;
  }

  setReasoningPromotion(enabled: boolean) {
    if (this.promoteReasoning === enabled) return;
    this.promoteReasoning = enabled;
    this.streamSnapshotValid = false;
  }

  items(finished: boolean): TranscriptItem[] {
    const lastAssistant = this.messages.map((message) => message.role).lastIndexOf("assistant");
    const items: TranscriptItem[] = [];
    const pending: { call: ToolCallValue; itemIndex?: number }[] = [];
    for (const [messageIndex, message] of this.messages.entries()) {
      if (message.role === "assistant") {
        const last = messageIndex === lastAssistant;
        for (const block of message.content) {
          if (block.type === "reasoning" && block.value.trim()) {
            if (last && this.promoteReasoning) {
              completeThinking(items);
              items.push({ type: "text", text: sanitize(block.value) });
            } else {
              const text = sanitize(block.value.trim());
              const previous = items.at(-1);
              if (previous?.type === "thinking") previous.text += `\n\n${text}`;
              else items.push({ type: "thinking", text, completed: false });
            }
          } else if (block.type === "text" && block.value.trim()) {
            completeThinking(items);
            items.push({ type: "text", text: sanitize(block.value) });
          } else if (block.type === "toolCall") {
            let itemIndex: number | undefined;
            if (!hiddenCall(block.call)) {
              completeThinking(items);
              itemIndex = items.length;
              items.push({ type: "toolCall", name: block.call.name, called: false });
            }
            // Hidden calls still consume their own results.
            pending.push({ call: block.call, itemIndex });
          }
        }
      } else if (message.role === "tool") {
        for (const block of message.content) {
          if (block.type !== "toolResult") continue;
          // IDs take precedence. When a template omits IDs, consume just one
          // matching name in call order, never every call.
          const exact =
            block.identifier !== undefined ? pending.findIndex(({ call }) => call.identifier === block.identifier) : -1;
          const matched =
            exact >= 0 || block.name === undefined
              ? exact
              : pending.findIndex(
                  ({ call }) =>
                    (block.identifier === undefined || call.identifier === undefined) && call.name === block.name,
                );
          if (matched < 0) continue;
          const [{ call, itemIndex }] = pending.splice(matched, 1) as [(typeof pending)[number]];
          if (itemIndex === undefined) continue;
          const result = parseJson(block.value);
          const failed = !!result && typeof result === "object" && "error" in result;
          items[itemIndex] =
            call.name === "show_chart" && !failed && isChartSpec(result)
              ? { type: "chart", chart: result }
              : {
                  type: "toolCall",
                  name: call.name,
                  called: true,
                  ...(failed || call.name === "show_chart" ? { failed: true } : {}),
                };
        }
      }
    }
    // ToolCalls starts another assistant turn. Hidden housekeeping creates no
    // visible boundary: leave thinking open until content resumes or the whole
    // run ends, including cancellation and errors.
    if (finished || (this.finishReason && this.finishReason !== "ToolCalls")) completeThinking(items);
    return items;
  }
}
