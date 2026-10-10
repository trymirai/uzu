import type { ReasoningSupport } from "./sampling";

export enum ModelKind {
  Text = "text",
}

export type UzuModel = {
  repoId: string;
  kind: ModelKind;
  name: string;
  vendor: string;
  reasoning: ReasoningSupport;
  supportsTools: boolean;
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
