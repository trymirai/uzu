import { getPlatform } from "@/platform/platform-singleton";
import { APP_STORE_KEY, migrateAppStorage } from "./migrate-app-storage";
import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ChatWidth = 768 | 1000 | 1200;
export const DEFAULT_CHAT_WIDTH: ChatWidth = 1000;

/*
 * Each chat width setting is a range, not a fixed number. Up to a content panel about 1500px wide
 * (any laptop) a setting is its plain width: Narrow 768, Medium 1000, Wide 1200. On bigger screens it
 * grows with the panel (cqw: the panel is a size container, see routes/__root.tsx) until its cap, so
 * a 32" monitor isn't mostly empty: there Narrow lands near the old Wide and Medium above it.
 */
const FLUID_CHAT_WIDTH: Record<ChatWidth, { share: number; max: number }> = {
  768: { share: 51.2, max: 1200 },
  1000: { share: 66.7, max: 1600 },
  1200: { share: 80, max: 2000 },
};

/** The chat column's width for a setting, as CSS: its base width, growing on large panels. */
export const chatWidthCss = (width: ChatWidth): string => {
  const { share, max } = FLUID_CHAT_WIDTH[width];
  return `clamp(${width}px, ${share}cqw, ${max}px)`;
};

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

const isChatWidth = (value: unknown): value is ChatWidth => value === 768 || value === 1000 || value === 1200;

// Narrow used to be 800px; it's 768px now (as in ChatGPT). A saved 800 stays Narrow.
const LEGACY_NARROW_WIDTH = 800;

type AppState = {
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
      theme: "system",
      isDarkMode: isDarkTheme("system"),
      chatWidth: DEFAULT_CHAT_WIDTH,
      appVersion: "",

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
        chatWidth: state.chatWidth,
      }),
      merge: (persisted, current) => {
        const stored = (persisted ?? {}) as Partial<Omit<AppState, "chatWidth">> & { chatWidth?: unknown };
        const theme = resolveTheme(stored);
        const chatWidth = stored.chatWidth === LEGACY_NARROW_WIDTH ? 768 : stored.chatWidth;
        return {
          ...current,
          theme,
          isDarkMode: isDarkTheme(theme),
          chatWidth: isChatWidth(chatWidth) ? chatWidth : DEFAULT_CHAT_WIDTH,
        };
      },
    },
  ),
);
