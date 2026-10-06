import type { SystemUiService } from ".";

export const webSystemUi: SystemUiService = {
  getRunOnStartup: () => Promise.resolve(false),
  setRunOnStartup: () => Promise.resolve(),
  getQuickEntryShortcut: () => Promise.resolve(null),
  registerQuickEntryShortcut: () => Promise.resolve(false),
  unregisterQuickEntryShortcut: () => Promise.resolve(),
  setWindowTheme: () => Promise.resolve(false),
};
