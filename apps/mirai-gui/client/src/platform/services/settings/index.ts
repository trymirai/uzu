import type { ModelParams } from "@/types/sampling";

export type SettingsService = {
  getEnableThinking(): Promise<boolean>;
  setEnableThinking(enabled: boolean): Promise<boolean>;
  getModelParams(): Promise<Record<string, ModelParams>>;
  setModelParams(repoId: string, params: ModelParams | null): Promise<void>;
  getAutoEjectEnabled(): Promise<boolean>;
  setAutoEjectEnabled(enabled: boolean): Promise<boolean>;
  getAutoEjectMinutes(): Promise<number>;
  setAutoEjectMinutes(minutes: number): Promise<boolean>;
};
