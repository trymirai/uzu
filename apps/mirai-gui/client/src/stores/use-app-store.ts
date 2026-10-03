import { getPlatform } from "@/platform/platform-singleton";
import { APP_STORE_KEY, migrateAppStorage } from "./migrate-app-storage";
import { create } from "zustand";
import { persist } from "zustand/middleware";

type AppState = {
  skipWelcome: boolean;
  completeWelcome: () => void;
  isDarkMode: boolean;
  setDarkMode: (isDark: boolean) => void;
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
      isDarkMode: true,
      appVersion: "",

      completeWelcome: () => set({ skipWelcome: true }),

      setDarkMode: (isDark) => {
        set({ isDarkMode: isDark });
        if (isDark) {
          document.documentElement.classList.add("dark");
        } else {
          document.documentElement.classList.remove("dark");
        }
        void getPlatform()
          .systemUi.setWindowTheme(isDark)
          .catch(() => {});
      },

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
        isDarkMode: state.isDarkMode,
        skipWelcome: state.skipWelcome,
      }),
    },
  ),
);
