import type { ReasoningSupport } from "@/types/sampling";

export enum ModelKind {
  Text = "text",
}

export type UzuModel = {
  repoId: string;
  kind: ModelKind;
  name: string;
  vendor: string;
  isThinking: boolean;
  reasoning?: ReasoningSupport;
  quantization?: string | null;
  quantizationBits?: number;
};

export type PlatformModel = UzuModel & {
  size?: number;
  family?: string;
  familyIdentifier?: string;
  paramSize?: number;
  sourceIndex?: number;
};
