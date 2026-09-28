import type { LlmAsyncStream, LlmRunParams } from "@/types/llm-stream";
import type { SamplingPolicyPayload } from "@/types/sampling";

export type TitleGenParams = {
  repoId: string;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
};

export type SamplingDefaults = Omit<Extract<SamplingPolicyPayload, { type: "Stochastic" }>, "type">;

export type ChatService = {
  runStream(params: LlmRunParams): LlmAsyncStream;
  cancelRun(runId: string): Promise<void>;
  generateTitle(params: TitleGenParams): Promise<string>;
  cancelTitleGen(): Promise<void>;
  /** Model's own generation defaults; null until that model's session is loaded. */
  getSamplingDefaults(repoId: string): Promise<SamplingDefaults | null>;
};
