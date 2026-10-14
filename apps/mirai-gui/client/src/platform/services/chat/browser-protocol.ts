import type { TitleGenParams } from ".";
import type { ReasoningSupport } from "@/types/sampling";
import type { LlmRunParams } from "@/types/llm-stream";
import type { RunEvent } from "./run-stream";
import type { SessionLoadingPayload, SessionStatePayload } from "../session";

export type BrowserEncodingSupport = { reasoning: ReasoningSupport; supportsTools: boolean };

export type EngineRequest =
  | { id: string; type: "inspectEncoding"; encoding: unknown }
  | { id: string; type: "inspectModelConfig"; config: Uint8Array }
  | { id: string; type: "title"; params: TitleGenParams; loadCancellation: Int32Array }
  | { id: string; type: "run"; params: LlmRunParams; loadCancellation: Int32Array }
  | { id: string; type: "cancel"; runId: string }
  | { id: string; type: "eject"; repoId: string };

export type EngineResponse =
  | { id: string; type: "event"; event: RunEvent }
  | { id: string; type: "result"; error?: string; value?: unknown }
  | { type: "session"; state: SessionStatePayload }
  | { type: "loading"; state: SessionLoadingPayload };
