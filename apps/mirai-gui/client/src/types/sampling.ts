export type SamplingPolicyPayload =
  | { type: "Default" }
  | { type: "Argmax" }
  | {
      type: "Stochastic";
      temperature?: number;
      topK?: number;
      topP?: number;
      minP?: number;
      repetitionPenalty?: number;
      suffixRepetitionLength?: number;
    };

export type ReasoningEffort = "disabled" | "default" | "low" | "medium" | "high" | "xhigh";

export const REASONING_EFFORTS: readonly ReasoningEffort[] = ["disabled", "default", "low", "medium", "high", "xhigh"];

export const isReasoningEffort = (value: unknown): value is ReasoningEffort =>
  typeof value === "string" && (REASONING_EFFORTS as readonly string[]).includes(value);

export type ReasoningSupport =
  { kind: "unsupported" } | { kind: "alwaysOn" } | { kind: "toggle" } | { kind: "levels"; efforts: ReasoningEffort[] };

export type ModelParams = {
  sampling: SamplingPolicyPayload;
  reasoningEffort?: ReasoningEffort;
};
