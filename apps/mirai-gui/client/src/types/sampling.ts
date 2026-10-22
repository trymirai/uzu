export type SamplingPolicyPayload =
  | { type: "Default" }
  | { type: "Greedy" }
  | {
      type: "Stochastic";
      temperature?: number | null;
      topK?: number | null;
      topP?: number | null;
      minP?: number | null;
      repetitionPenalty?: number | null;
      suffixRepetitionLength?: number | null;
    };

// Same floor as the CLI.
export const TEMPERATURE_MIN = 0.05;

export const normalizeSampling = (sampling: SamplingPolicyPayload | { type: "Argmax" }): SamplingPolicyPayload =>
  sampling.type === "Argmax" ||
  (sampling.type === "Stochastic" && sampling.temperature != null && sampling.temperature <= 0)
    ? { type: "Greedy" }
    : sampling;

export type ReasoningEffort = "disabled" | "default" | "low" | "medium" | "high" | "xhigh";

export const REASONING_EFFORTS: readonly ReasoningEffort[] = ["disabled", "default", "low", "medium", "high", "xhigh"];

export const isReasoningEffort = (value: unknown): value is ReasoningEffort =>
  typeof value === "string" && (REASONING_EFFORTS as readonly string[]).includes(value);

export type ReasoningSupport =
  | { kind: "unsupported" }
  | { kind: "alwaysOn" }
  | { kind: "toggle"; defaultEffort: "default" }
  | {
      kind: "levels";
      efforts: Exclude<ReasoningEffort, "default">[];
      defaultEffort?: Exclude<ReasoningEffort, "default">;
    };

export type ModelParams = {
  sampling: SamplingPolicyPayload;
  reasoningEffort?: ReasoningEffort;
  modelChatNamingEnabled?: boolean;
  dateTimeToolEnabled?: boolean;
  chartToolEnabled?: boolean;
};
