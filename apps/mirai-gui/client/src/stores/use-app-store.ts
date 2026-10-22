import { getPlatform } from "@/platform/platform-singleton";
import { APP_STORE_KEY, migrateAppStorage } from "./migrate-app-storage";
import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ChatWidth = 800 | 1000 | 1200;
export const DEFAULT_CHAT_WIDTH: ChatWidth = 1000;

export type ThemeMode = "system" | "light" | "dark";

const resolveTheme = (stored: { theme?: unknown; isDarkMode?: unknown }): ThemeMode =>
  stored.theme === "system" || stored.theme === "light" || stored.theme === "dark"
    ? stored.theme
    : stored.isDarkMode === false
      ? "light"
      : "system";

export const isDarkTheme = (
  theme: ThemeMode,
  systemDark = typeof window !== "undefined" && (window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false),
): boolean => theme === "dark" || (theme === "system" && systemDark);

const isChatWidth = (value: unknown): value is ChatWidth => value === 800 || value === 1000 || value === 1200;

type AppState = {
  skipWelcome: boolean;
  completeWelcome: () => void;
  isDarkMode: boolean;
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  chatWidth: ChatWidth;
  setChatWidth: (width: number) => void;
  appVersion: string;
  fetchVersion(): Promise<void>;
};

// Runs before create(persist(...)) so hydration sees the migrated key regardless
// of module order.
migrateAppStorage();

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      skipWelcome: false,
      theme: "system",
      isDarkMode: isDarkTheme("system"),
      chatWidth: DEFAULT_CHAT_WIDTH,
      appVersion: "",

      completeWelcome: () => set({ skipWelcome: true }),
      setChatWidth: (width) => set({ chatWidth: isChatWidth(width) ? width : DEFAULT_CHAT_WIDTH }),

      setTheme: (theme) => set({ theme, isDarkMode: isDarkTheme(theme) }),

      fetchVersion: async () => {
        try {
          const version = await getPlatform().system.getAppVersion();
          if (version) set({ appVersion: version });
        } catch (error) {
          console.error("Failed to fetch app version:", error);
        }
      },
    }),
    {
      name: APP_STORE_KEY,
      partialize: (state) => ({
        theme: state.theme,
        skipWelcome: state.skipWelcome,
        chatWidth: state.chatWidth,
      }),
      merge: (persisted, current) => {
        const stored = (persisted ?? {}) as Partial<Omit<AppState, "chatWidth">> & { chatWidth?: unknown };
        const theme = resolveTheme(stored);
        const { chatWidth } = stored;
        return {
          ...current,
          skipWelcome: stored.skipWelcome ?? current.skipWelcome,
          theme,
          isDarkMode: isDarkTheme(theme),
          chatWidth: isChatWidth(chatWidth) ? chatWidth : DEFAULT_CHAT_WIDTH,
        };
      },
    },
  ),
);
