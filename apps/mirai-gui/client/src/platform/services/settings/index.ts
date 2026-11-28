import type { ModelParams } from "@/types/sampling";
import type { WebSearchKeyProvider, WebSearchProvider } from "@/constants/web-search-providers";
import type { WebSearchMode, WebSearchStoredKey } from "@/types/web-search";

export const DEFAULT_AUTO_EJECT_MINUTES = 15;

export type WebSearchModes = { chats: Record<string, WebSearchMode>; last: WebSearchMode };

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
  getWebSearchProvider(): Promise<WebSearchProvider>;
  setWebSearchProvider(provider: WebSearchProvider): Promise<void>;
  getWebSearchModes(): Promise<WebSearchModes>;
  setWebSearchModes(modes: WebSearchModes): Promise<void>;
  getWebSearchKeys(): Promise<WebSearchStoredKey[]>;
  saveWebSearchKey(provider: WebSearchKeyProvider, key: string): Promise<WebSearchStoredKey[]>;
  removeWebSearchKey(provider: WebSearchKeyProvider): Promise<WebSearchStoredKey[]>;
};
