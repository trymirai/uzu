import type { ChatRole } from "@/types/chat";
import { isChartSpec, type ChartSpec } from "@/types/chart";
import type { ReasoningEffort, SamplingPolicyPayload } from "@/types/sampling";
import {
  isRecord,
  isWebPageItem,
  isWebSearchItem,
  type SearchPermissionRequest,
  type WebPageItem,
  type WebSearchItem,
} from "@/types/web-search";

export type ReadPage = { url: string; readFrom?: string };

export type LlmRunParams = {
  repoId: string;
  chatId: string;
  messages: Array<{
    role: ChatRole;
    content: string;
    reasoningContent?: string;
    openableUrls?: string[];
    readPages?: ReadPage[];
  }>;
  samplingPolicy?: SamplingPolicyPayload;
  reasoningEffort?: ReasoningEffort;
  modelChatNamingEnabled?: boolean;
  dateTimeToolEnabled?: boolean;
  chartToolEnabled?: boolean;
  webSearchEnabled?: boolean;
  webSearchAsk?: boolean;
};

export type SessionOutputStats = {
  tokensPerForwardPass?: number;
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
  | { type: "toolCall"; name: string; called: boolean; failed?: boolean }
  | WebSearchItem
  | WebPageItem;

export const isTranscript = (value: unknown): value is TranscriptItem[] =>
  Array.isArray(value) &&
  value.every((item: unknown) => {
    if (!isRecord(item)) return false;
    const { type, text, completed, name, called, failed, chart } = item;
    switch (type) {
      case "thinking":
        return typeof text === "string" && (completed === undefined || typeof completed === "boolean");
      case "text":
        return typeof text === "string";
      case "chart":
        return isChartSpec(chart);
      case "toolCall":
        return (
          typeof name === "string" &&
          typeof called === "boolean" &&
          (failed === undefined || typeof failed === "boolean")
        );
      case "webSearch":
        return isWebSearchItem(item);
      case "webPage":
        return isWebPageItem(item);
      default:
        return false;
    }
  });

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
  onSearchPermission: (cb: (request: SearchPermissionRequest) => void) => () => void;
};
