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

// Same floor as the CLI.
export const TEMPERATURE_MIN = 0.05;

export const normalizeSampling = (sampling: SamplingPolicyPayload): SamplingPolicyPayload =>
  sampling.type === "Stochastic" && sampling.temperature !== undefined && sampling.temperature <= 0
    ? { type: "Argmax" }
    : sampling;

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
