import type { ModelParams } from "@/types/sampling";
import { DEFAULT_AUTO_EJECT_MINUTES, type SettingsService } from ".";
import { readJson, writeJson } from "../shared/browser-store";

const SETTINGS_KEY = "mirai.web.settings";
const MODEL_PARAMS_KEY = "mirai.web.modelParams";

type StoredSettings = {
  analyticsEnabled: boolean;
  modelChatNamingEnabled: boolean;
  autoEjectEnabled: boolean;
  autoEjectMinutes: number;
};

const DEFAULTS: StoredSettings = {
  analyticsEnabled: false,
  modelChatNamingEnabled: true,
  autoEjectEnabled: true,
  autoEjectMinutes: DEFAULT_AUTO_EJECT_MINUTES,
};

const load = (): StoredSettings => ({ ...DEFAULTS, ...readJson<Partial<StoredSettings>>(SETTINGS_KEY, {}) });
const mutate = (key: string, run: () => void): Promise<void> =>
  navigator.locks ? navigator.locks.request(key, run) : Promise.resolve().then(run);
const autoEjectListeners = new Set<() => void>();
export function onBrowserAutoEjectChanged(listener: () => void): () => void {
  autoEjectListeners.add(listener);
  const externalChange = (event: StorageEvent) => {
    if (event.key === SETTINGS_KEY || event.key === null) listener();
  };
  window.addEventListener("storage", externalChange);
  return () => {
    autoEjectListeners.delete(listener);
    window.removeEventListener("storage", externalChange);
  };
}

export const webSettings: SettingsService = {
  getAnalyticsEnabled: async () => false,

  setAnalyticsEnabled: async (enabled) => {
    if (enabled) throw new Error("Usage analytics is unavailable in the browser");
    await mutate(SETTINGS_KEY, () => writeJson(SETTINGS_KEY, { ...load(), analyticsEnabled: false }));
  },

  getModelChatNamingEnabled: async () => load().modelChatNamingEnabled !== false,

  setModelChatNamingEnabled: (enabled) =>
    mutate(SETTINGS_KEY, () => writeJson(SETTINGS_KEY, { ...load(), modelChatNamingEnabled: enabled })),

  getModelParams: async () => readJson<Record<string, ModelParams>>(MODEL_PARAMS_KEY, {}),

  setModelParams: (repoId, params) =>
    mutate(MODEL_PARAMS_KEY, () => {
      const all = readJson<Record<string, ModelParams>>(MODEL_PARAMS_KEY, {});
      if (params === null) delete all[repoId];
      else all[repoId] = params;
      writeJson(MODEL_PARAMS_KEY, all);
    }),

  getAutoEjectEnabled: async () => load().autoEjectEnabled !== false,
  setAutoEjectEnabled: async (enabled) => {
    await mutate(SETTINGS_KEY, () => writeJson(SETTINGS_KEY, { ...load(), autoEjectEnabled: enabled }));
    autoEjectListeners.forEach((listener) => listener());
    return true;
  },
  getAutoEjectMinutes: async () => {
    const minutes = load().autoEjectMinutes;
    return typeof minutes === "number" && Number.isFinite(minutes) && minutes > 0
      ? minutes
      : DEFAULT_AUTO_EJECT_MINUTES;
  },
  setAutoEjectMinutes: async (minutes) => {
    if (!Number.isFinite(minutes) || minutes <= 0) throw new Error("Auto-eject interval must be positive");
    await mutate(SETTINGS_KEY, () => writeJson(SETTINGS_KEY, { ...load(), autoEjectMinutes: minutes }));
    autoEjectListeners.forEach((listener) => listener());
    return true;
  },
};
