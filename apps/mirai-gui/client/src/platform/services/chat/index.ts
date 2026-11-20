import type { LlmAsyncStream, LlmRunParams } from "@/types/llm-stream";
import type { SamplingPolicyPayload } from "@/types/sampling";
import type { SearchPermissionDecision } from "@/types/web-search";

export type TitleGenParams = {
  repoId: string;
  userText: string;
};

export type SamplingDefaults = Exclude<SamplingPolicyPayload, { type: "Default" }>;

export type ChatService = {
  runStream(params: LlmRunParams): LlmAsyncStream;
  cancelRun(runId: string): Promise<void>;
  answerSearchPermission(requestId: string, shownItems: number, decision: SearchPermissionDecision): Promise<boolean>;
  siteIcon(host: string): Promise<string | null>;
  generateTitle(params: TitleGenParams): Promise<string>;
  cancelTitleGen(): Promise<void>;
  /** Model's actual sampling method and parameters, read without loading weights. */
  getSamplingDefaults(repoId: string): Promise<SamplingDefaults | null>;
};
