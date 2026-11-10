import type {
  ChatMessage,
  ChatReply,
  ChatReplyStats,
  ChatSessionError,
  ReasoningEffort as UzuReasoningEffort,
  SamplingPolicy,
} from "@uzu/wasm";
import type { SessionOutputFinishReason } from "@/types/llm-stream";
import type { ReasoningEffort, SamplingPolicyPayload } from "@/types/sampling";
import type { UzuBindings } from "../uzu-runtime";
import type { HistoryMessage } from "./text";
import type { ContentBlock, Reply, ReplyStats, TranscriptMessage } from "./transcript";

// Getters copy values out of wasm; release each copy once it has been read.
const release = (...values: unknown[]) => {
  for (const value of values) (value as { free?: () => void } | null | undefined)?.free?.();
};

const optional = <T>(value: T | null | undefined): T | undefined => value ?? undefined;

export const toUzuEffort = (effort: ReasoningEffort): UzuReasoningEffort =>
  (effort === "xhigh" ? "XHigh" : effort[0]!.toUpperCase() + effort.slice(1)) as UzuReasoningEffort;

function readBlock(api: UzuBindings, block: unknown): ContentBlock {
  try {
    if (block instanceof api.ChatContentBlockText) return { type: "text", value: block.value };
    if (block instanceof api.ChatContentBlockReasoning) return { type: "reasoning", value: block.value };
    if (block instanceof api.ChatContentBlockToolCall) {
      const call = block.value;
      const args = call.arguments;
      try {
        return {
          type: "toolCall",
          call: {
            ...(call.identifier != null ? { identifier: call.identifier } : {}),
            name: call.name,
            arguments: args.json,
          },
        };
      } finally {
        release(args, call);
      }
    }
    if (block instanceof api.ChatContentBlockToolCallResult) {
      const value = block.value;
      try {
        return {
          type: "toolResult",
          ...(block.identifier != null ? { identifier: block.identifier } : {}),
          ...(block.name != null ? { name: block.name } : {}),
          value: value.json,
        };
      } finally {
        release(value);
      }
    }
    return { type: "other" };
  } finally {
    release(block);
  }
}

export function readMessage(api: UzuBindings, message: ChatMessage): TranscriptMessage {
  const role = message.role;
  const metadata = message.metadata;
  const values = metadata.values;
  try {
    return {
      role: role instanceof api.ChatRoleAssistant ? "assistant" : role instanceof api.ChatRoleTool ? "tool" : "other",
      metadata: JSON.stringify(
        Object.entries(values)
          .map(([key, value]) => [key, value.json])
          .sort(([a], [b]) => (a! < b! ? -1 : 1)),
      ),
      content: message.content.map((block) => readBlock(api, block)),
    };
  } finally {
    release(role, ...Object.values(values), metadata);
  }
}

function readStats(stats: ChatReplyStats): ReplyStats {
  const speculator = stats.speculatorStats;
  try {
    return {
      duration: stats.duration,
      timeToFirstToken: optional(stats.timeToFirstToken),
      prefillTokensPerSecond: optional(stats.prefillTokensPerSecond),
      generateTokensPerSecond: optional(stats.generateTokensPerSecond),
      tokensCountInput: optional(stats.tokensCountInput),
      tokensCountOutput: optional(stats.tokensCountOutput),
      tokensPerForwardPass: optional(speculator?.tokensPerForwardPass),
    };
  } finally {
    release(speculator, stats);
  }
}

export function readReply(api: UzuBindings, reply: ChatReply): Reply {
  const message = reply.message;
  try {
    return {
      message: readMessage(api, message),
      finishReason: optional(reply.finishReason) as SessionOutputFinishReason | undefined,
      stats: readStats(reply.stats),
    };
  } finally {
    release(message, reply);
  }
}

function historyMessage(api: UzuBindings, message: HistoryMessage): ChatMessage {
  if (message.role === "system") return api.ChatMessage.system().withText(message.content);
  if (message.role === "user") return api.ChatMessage.user().withText(message.content);
  const assistant = api.ChatMessage.assistant();
  return (message.reasoningContent ? assistant.withReasoning(message.reasoningContent) : assistant).withText(
    message.content,
  );
}

// uzu reads reasoning effort from a system-message block.
export function buildMessages(
  api: UzuBindings,
  history: HistoryMessage[],
  effort: ReasoningEffort | undefined,
): ChatMessage[] {
  const messages = history.map((message) => historyMessage(api, message));
  if (effort) {
    const system = history.findIndex((message) => message.role === "system");
    if (system >= 0) messages[system] = messages[system]!.withReasoningEffort(toUzuEffort(effort));
    else messages.unshift(api.ChatMessage.system().withReasoningEffort(toUzuEffort(effort)));
  }
  return messages;
}

// uzu rejects another system message in an accumulated session.
export const buildTailMessages = (api: UzuBindings, tail: HistoryMessage[]): ChatMessage[] =>
  tail.filter((message) => message.role !== "system").map((message) => historyMessage(api, message));

export function addNamingInstruction(api: UzuBindings, messages: ChatMessage[], instruction: string) {
  const system = messages.findIndex((message) => message.role instanceof api.ChatRoleSystem);
  if (system >= 0) messages[system] = messages[system]!.withText(`\n\n${instruction}`);
  else messages.unshift(api.ChatMessage.system().withText(instruction));
}

export function samplingPolicy(api: UzuBindings, payload: SamplingPolicyPayload | undefined): SamplingPolicy {
  if (!payload || payload.type === "Default") return new api.SamplingPolicyDefault();
  if (payload.type === "Greedy" || (payload.temperature != null && payload.temperature <= 0))
    return new api.SamplingPolicyCustom(new api.SamplingMethodGreedy());
  return new api.SamplingPolicyCustom(
    new api.SamplingMethodStochastic(
      payload.temperature,
      payload.topK,
      payload.topP,
      payload.minP,
      payload.repetitionPenalty,
      // uzu panics on a penalty without a suffix length (and on 0).
      payload.repetitionPenalty != null
        ? payload.suffixRepetitionLength != null && payload.suffixRepetitionLength > 0
          ? payload.suffixRepetitionLength
          : 32
        : null,
    ),
  );
}

const BACKEND_ERRORS: Record<string, string> = {
  Backend: "Backend error",
  Capture: "Capture error",
  Decoder: "Decoder error",
  Grammar: "Grammar error",
  TrieAccept: "Trie accept error",
  Speculator: "Speculator error",
};

// The same messages as the Display implementations in Rust.
export function chatErrorMessage(error: ChatSessionError | Error): string {
  if (error instanceof Error) return error.message;
  switch (error.type) {
    case "Backend": {
      const backend = error.error;
      if (backend.type === "ContextOverflow") return "Conversation is too long for this model's context window.";
      if (backend.type === "NoSeedToken") return "Backend error: No seed token (both state and input are empty)";
      const prefix = BACKEND_ERRORS[backend.type];
      return `Backend error: ${prefix ? `${prefix}: ${backend.message}` : backend.message}`;
    }
    case "Loading":
      return `Unable to load model: ${error.message}`;
    case "UnsupportedModel":
      return "Unsupported model";
    case "UnableToPerformOperationInCurrentState":
      return "Unable to perform operation in current state";
    case "NoResponse":
      return "No response";
    case "ToolTurnLimitExceeded":
      return `Tool turn limit exceeded: ${error.limit}`;
  }
}
