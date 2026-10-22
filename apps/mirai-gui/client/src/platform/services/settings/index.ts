import type { ModelParams } from "@/types/sampling";

export const DEFAULT_AUTO_EJECT_MINUTES = 15;

export type SettingsService = {
  getAnalyticsEnabled(): Promise<boolean>;
  setAnalyticsEnabled(enabled: boolean): Promise<void>;
  getModelChatNamingEnabled(): Promise<boolean>;
  setModelChatNamingEnabled(enabled: boolean): Promise<void>;
  getModelParams(): Promise<Record<string, ModelParams>>;
  setModelParams(repoId: string, params: ModelParams | null): Promise<void>;
  getAutoEjectEnabled(): Promise<boolean>;
  setAutoEjectEnabled(enabled: boolean): Promise<boolean>;
  getAutoEjectMinutes(): Promise<number>;
  setAutoEjectMinutes(minutes: number): Promise<boolean>;
};
