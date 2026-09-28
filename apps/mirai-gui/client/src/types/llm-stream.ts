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

export type RunMetrics = {
  ttftSec?: number;
  totalSec?: number;
  tps?: number;
  tokensOut: number;
};

export function getRunMetrics(stats: SessionOutputStats): RunMetrics {
  return {
    ttftSec: stats.prefillStats.duration,
    totalSec: stats.totalStats.duration,
    tps: stats.generateStats?.tokensPerSecond,
    tokensOut: stats.totalStats.tokensCountOutput,
  };
}

export type ParsedPatch = { response?: string; chainOfThought?: string };
export type OutputShape = { text?: { parsed?: ParsedPatch; raw?: string } };

export function withParsedOutput(output: OutputShape | undefined, patch: ParsedPatch): OutputShape {
  return {
    ...(output || {}),
    text: {
      ...(output?.text || {}),
      parsed: {
        ...(output?.text?.parsed || {}),
        ...(patch.chainOfThought !== undefined ? { chainOfThought: patch.chainOfThought } : {}),
        ...(patch.response !== undefined ? { response: patch.response } : {}),
      },
    },
  };
}

export type SessionOutputFinishReason =
  | "Stop"
  | "Length"
  | "Cancelled"
  | "ContextLimitReached"
  | "ToolCalls"
  | "Rejected";

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
