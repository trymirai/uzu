import { create } from "zustand";
import { getPlatform } from "@/platform/platformSingleton";
import { useModelParamsStore } from "@/stores/useModelParamsStore";

type SettingsState = {
  enableThinking: boolean;
  runOnStartup: boolean;
  quickEntryShortcut: string | null;
  autoEjectEnabled: boolean;
  autoEjectMinutes: number;

  fetch: () => Promise<void>;
  fetchDesktopSettings: () => Promise<void>;
  setEnableThinking: (enabled: boolean) => Promise<void>;
  setRunOnStartup: (value: boolean) => Promise<void>;
  registerQuickEntryShortcut: (accelerator: string) => Promise<boolean>;
  unregisterQuickEntryShortcut: () => Promise<void>;
  setAutoEjectEnabled: (value: boolean) => Promise<void>;
  setAutoEjectMinutes: (minutes: number) => Promise<void>;
  exportLogs: () => Promise<"ok" | "cancelled" | "error">;
};

export const useSettingsStore = create<SettingsState>((set, get) => ({
  enableThinking: true,
  runOnStartup: false,
  quickEntryShortcut: null,
  autoEjectEnabled: true,
  autoEjectMinutes: 2,

  fetch: async () => {
    const { settings } = getPlatform();
    const [thinking, aeEnabled, aeMinutes] = await Promise.all([
      settings.getEnableThinking().catch(() => true),
      settings.getAutoEjectEnabled().catch(() => true),
      settings.getAutoEjectMinutes().catch(() => 2),
    ]);
    set({
      enableThinking: thinking,
      autoEjectEnabled: aeEnabled,
      autoEjectMinutes: Number.isFinite(aeMinutes) ? aeMinutes : 2,
    });
  },

  fetchDesktopSettings: async () => {
    const { systemUi } = getPlatform();
    const [startup, shortcut] = await Promise.all([
      systemUi.getRunOnStartup().catch(() => false),
      systemUi.getQuickEntryShortcut().catch(() => null),
    ]);
    set({ runOnStartup: startup, quickEntryShortcut: shortcut });
  },

  setEnableThinking: async (enabled) => {
    const { settings } = getPlatform();
    set({ enableThinking: enabled });
    useModelParamsStore.getState().setGlobalReasoningEnabled(enabled);
    await settings.setEnableThinking(enabled).catch(() => {
      set({ enableThinking: !enabled });
      useModelParamsStore.getState().setGlobalReasoningEnabled(!enabled);
    });
  },

  setRunOnStartup: async (value) => {
    set({ runOnStartup: value });
    await getPlatform()
      .systemUi.setRunOnStartup(value)
      .catch(() => set({ runOnStartup: !value }));
  },

  registerQuickEntryShortcut: async (accelerator) => {
    const ok = await getPlatform()
      .systemUi.registerQuickEntryShortcut(accelerator)
      .catch(() => false);
    if (ok) set({ quickEntryShortcut: accelerator });
    return ok;
  },

  unregisterQuickEntryShortcut: async () => {
    try {
      await getPlatform().systemUi.unregisterQuickEntryShortcut();
      set({ quickEntryShortcut: null });
    } catch (e) {
      console.error("[settings] unregisterQuickEntryShortcut failed", e);
    }
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
