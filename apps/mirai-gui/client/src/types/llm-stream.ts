import type { ChatRole } from "@/types/chat";
import type { ReasoningEffort, SamplingPolicyPayload } from "@/types/sampling";

export type LlmRunParams = {
  repoId: string;
  messages: Array<{ role: ChatRole; content: string; reasoningContent?: string }>;
  samplingPolicy?: SamplingPolicyPayload;
  reasoningEffort?: ReasoningEffort;
};

export type SessionOutputStats = {
  prefillStats: {
    duration: number;
    tokensCount: number;
    tokensPerSecond: number;
  };
  generateStats?: {
    duration: number;
    tokensCount: number;
    tokensPerSecond: number;
  };
  totalStats: {
    duration: number;
    tokensCountInput: number;
    tokensCountOutput: number;
  };
};

export type ParsedPatch = { response?: string; chainOfThought?: string };
export type OutputShape = { text?: { parsed?: ParsedPatch; raw?: string } };

export type SessionOutputFinishReason =
  "Stop" | "Length" | "Cancelled" | "ContextLimitReached" | "ToolCalls" | "Rejected";

export type LlmRunResult = {
  text: string;
  stats: SessionOutputStats;
  finishReason?: SessionOutputFinishReason;
  parsed?: { chainOfThought?: string; response?: string };
  error?: string;
};

export type LlmAsyncStream = {
  runId: string;
  stream: ReadableStream<string>;
  result: Promise<LlmRunResult>;
  cancel: () => Promise<void>;
  onParsed: (cb: (p: { chainOfThought?: string; response?: string }) => void) => () => void;
};
