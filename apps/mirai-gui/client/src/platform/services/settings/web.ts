import type { ModelParams } from "@/types/sampling";
import { DEFAULT_AUTO_EJECT_MINUTES, type SettingsService } from ".";
import { readJson, writeJson } from "../shared/browser-store";

const SETTINGS_KEY = "mirai.web.settings";
const MODEL_PARAMS_KEY = "mirai.web.modelParams";

type StoredSettings = { analyticsEnabled: boolean; modelChatNamingEnabled: boolean };

const DEFAULTS: StoredSettings = { analyticsEnabled: false, modelChatNamingEnabled: true };

const load = (): StoredSettings => ({ ...DEFAULTS, ...readJson<Partial<StoredSettings>>(SETTINGS_KEY, {}) });

export const webSettings: SettingsService = {
  getAnalyticsEnabled: async () => load().analyticsEnabled === true,

  setAnalyticsEnabled: async (enabled) => {
    writeJson(SETTINGS_KEY, { ...load(), analyticsEnabled: enabled });
  },

  getModelChatNamingEnabled: async () => load().modelChatNamingEnabled !== false,

  setModelChatNamingEnabled: async (enabled) => {
    writeJson(SETTINGS_KEY, { ...load(), modelChatNamingEnabled: enabled });
  },

  getModelParams: async () => readJson<Record<string, ModelParams>>(MODEL_PARAMS_KEY, {}),

  setModelParams: async (repoId, params) => {
    const all = readJson<Record<string, ModelParams>>(MODEL_PARAMS_KEY, {});
    if (params === null) delete all[repoId];
    else all[repoId] = params;
    writeJson(MODEL_PARAMS_KEY, all);
  },

  // No resident engine in the browser, so auto-eject is inert.
  getAutoEjectEnabled: async () => true,
  setAutoEjectEnabled: async (enabled) => enabled,
  getAutoEjectMinutes: async () => DEFAULT_AUTO_EJECT_MINUTES,
  setAutoEjectMinutes: async () => true,
};
