import type { ModelParams } from "@/types/sampling";
import type { SettingsService } from ".";
import { readJson, writeJson } from "../shared/browser-store";

const SETTINGS_KEY = "mirai.web.settings";
const MODEL_PARAMS_KEY = "mirai.web.modelParams";

type StoredSettings = { enableThinking: boolean };

const DEFAULTS: StoredSettings = { enableThinking: false };

const load = (): StoredSettings => ({ ...DEFAULTS, ...readJson<Partial<StoredSettings>>(SETTINGS_KEY, {}) });

export const webSettings: SettingsService = {
  getEnableThinking: async () => load().enableThinking,

  setEnableThinking: async (enabled) => {
    writeJson(SETTINGS_KEY, { ...load(), enableThinking: enabled });
    return enabled;
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
  getAutoEjectMinutes: async () => 2,
  setAutoEjectMinutes: async () => true,
};
