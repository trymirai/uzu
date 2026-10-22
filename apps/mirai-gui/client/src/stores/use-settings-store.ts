import { create } from "zustand";
import { getPlatform } from "@/platform/platform-singleton";
import { DEFAULT_AUTO_EJECT_MINUTES } from "@/platform/services/settings";
import { useModelParamsStore } from "./use-model-params-store";

type SettingsState = {
  analyticsEnabled: boolean;
  modelChatNamingEnabled: boolean;
  autoEjectEnabled: boolean;
  autoEjectMinutes: number;

  fetch: () => Promise<void>;
  setAnalyticsEnabled: (enabled: boolean) => Promise<void>;
  setModelChatNamingEnabled: (enabled: boolean) => Promise<void>;
  setAutoEjectEnabled: (value: boolean) => Promise<void>;
  setAutoEjectMinutes: (minutes: number) => Promise<void>;
  exportLogs: () => Promise<"ok" | "cancelled" | "error">;
};

export const useSettingsStore = create<SettingsState>((set, get) => ({
  analyticsEnabled: false,
  modelChatNamingEnabled: true,
  autoEjectEnabled: true,
  autoEjectMinutes: DEFAULT_AUTO_EJECT_MINUTES,

  fetch: async () => {
    const { settings } = getPlatform();
    const [analyticsEnabled, modelChatNamingEnabled, aeEnabled, aeMinutes] = await Promise.all([
      settings.getAnalyticsEnabled().catch(() => false),
      settings.getModelChatNamingEnabled().catch(() => true),
      settings.getAutoEjectEnabled().catch(() => true),
      settings.getAutoEjectMinutes().catch(() => DEFAULT_AUTO_EJECT_MINUTES),
    ]);
    set({
      analyticsEnabled,
      modelChatNamingEnabled,
      autoEjectEnabled: aeEnabled,
      autoEjectMinutes: Number.isFinite(aeMinutes) && aeMinutes > 0 ? aeMinutes : DEFAULT_AUTO_EJECT_MINUTES,
    });
    useModelParamsStore.getState().setGlobalModelChatNamingEnabled(modelChatNamingEnabled);
  },

  setAnalyticsEnabled: async (enabled) => {
    const previous = get().analyticsEnabled;
    await getPlatform()
      .settings.setAnalyticsEnabled(enabled)
      .then(() => set({ analyticsEnabled: enabled }))
      .catch(() => set({ analyticsEnabled: previous }));
  },

  setModelChatNamingEnabled: async (enabled) => {
    const previous = get().modelChatNamingEnabled;
    set({ modelChatNamingEnabled: enabled });
    useModelParamsStore.getState().setGlobalModelChatNamingEnabled(enabled);
    await getPlatform()
      .settings.setModelChatNamingEnabled(enabled)
      .catch(() => {
        set({ modelChatNamingEnabled: previous });
        useModelParamsStore.getState().setGlobalModelChatNamingEnabled(previous);
      });
  },

  setAutoEjectEnabled: async (value) => {
    set({ autoEjectEnabled: value });
    await getPlatform()
      .settings.setAutoEjectEnabled(value)
      .catch(() => set({ autoEjectEnabled: !value }));
  },

  setAutoEjectMinutes: async (minutes) => {
    const prev = get().autoEjectMinutes;
    set({ autoEjectMinutes: minutes });
    await getPlatform()
      .settings.setAutoEjectMinutes(minutes)
      .catch(() => set({ autoEjectMinutes: prev }));
  },

  exportLogs: async () => {
    try {
      const { system, dialogs, storage } = getPlatform();
      const logFilePath = await system.getLogFilePath();
      if (!logFilePath) return "error";
      const content = await dialogs.readTextFile(logFilePath);
      if (content === null) return "error";
      const destination = await dialogs.showSaveDialog({
        title: "Export logs",
        defaultPath: "mirai.log",
        filters: [{ name: "Log files", extensions: ["log", "txt"] }],
      });
      if (!destination) return "cancelled";
      const saved = await storage.saveBinaryFile(destination, new TextEncoder().encode(content));
      return saved ? "ok" : "error";
    } catch {
      return "error";
    }
  },
}));
