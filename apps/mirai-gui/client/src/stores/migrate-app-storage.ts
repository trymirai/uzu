export const APP_STORE_KEY = "mirai-store";

// Keys written by earlier releases. The names are fixed by what is already in
// users' browser storage, so they are read verbatim and never renamed.
const LEGACY_APP_STORE_KEY = "mirai-electron-store";
const LEGACY_AUTH_STORE_KEY = "mirai-auth-store";

const readState = (key: string): Record<string, unknown> | null => {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const state = parsed?.state;
    return state && typeof state === "object" ? (state as Record<string, unknown>) : null;
  } catch {
    return null;
  }
};

const migrateAppStorage = (): void => {
  try {
    if (localStorage.getItem(APP_STORE_KEY)) {
      localStorage.removeItem(LEGACY_APP_STORE_KEY);
      localStorage.removeItem(LEGACY_AUTH_STORE_KEY);
      return;
    }

    const legacyApp = readState(LEGACY_APP_STORE_KEY);
    const legacyAuth = readState(LEGACY_AUTH_STORE_KEY);

    const state: Record<string, unknown> = {};
    if (typeof legacyApp?.isDarkMode === "boolean") state.isDarkMode = legacyApp.isDarkMode;
    const skipWelcome = legacyApp?.skipWelcome ?? legacyAuth?.skipWelcome;
    if (typeof skipWelcome === "boolean") state.skipWelcome = skipWelcome;

    if (Object.keys(state).length > 0) {
      localStorage.setItem(APP_STORE_KEY, JSON.stringify({ state, version: 0 }));
    }
    localStorage.removeItem(LEGACY_APP_STORE_KEY);
    localStorage.removeItem(LEGACY_AUTH_STORE_KEY);
  } catch {
    // Boot must survive a blocked or full localStorage.
  }
};

export { migrateAppStorage };
