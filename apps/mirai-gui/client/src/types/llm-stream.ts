import type { ChatRole } from "@/types/chat";
import type { ChartSpec } from "@/types/chart";
import type { ReasoningEffort, SamplingPolicyPayload } from "@/types/sampling";

export type LlmRunParams = {
  repoId: string;
  messages: Array<{ role: ChatRole; content: string; reasoningContent?: string }>;
  samplingPolicy?: SamplingPolicyPayload;
  reasoningEffort?: ReasoningEffort;
  modelChatNamingEnabled?: boolean;
  dateTimeToolEnabled?: boolean;
  chartToolEnabled?: boolean;
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
export type TranscriptItem =
  | { type: "thinking"; text: string; completed?: boolean }
  | { type: "text"; text: string }
  | { type: "chart"; chart: ChartSpec }
  | { type: "toolCall"; name: string; called: boolean; failed?: boolean };

export type OutputShape = { text?: { parsed?: ParsedPatch; raw?: string }; transcript?: TranscriptItem[] };

export type SessionOutputFinishReason =
  "Stop" | "Length" | "Cancelled" | "ContextLimitReached" | "ToolCalls" | "Rejected";

export type LlmRunResult = {
  text: string;
  chatName?: string;
  stats: SessionOutputStats;
  finishReason?: SessionOutputFinishReason;
  parsed?: { chainOfThought?: string; response?: string };
  transcript?: TranscriptItem[];
  error?: string;
};

export type LlmAsyncStream = {
  runId: string;
  stream: ReadableStream<string>;
  result: Promise<LlmRunResult>;
  cancel: () => Promise<void>;
  onParsed: (cb: (p: { chainOfThought?: string; response?: string }) => void) => () => void;
  onChatName: (cb: (name: string) => void) => () => void;
  onTranscript: (cb: (items: TranscriptItem[]) => void) => () => void;
};
