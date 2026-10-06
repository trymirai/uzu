export type SystemUiService = {
  getRunOnStartup(): Promise<boolean>;
  setRunOnStartup(value: boolean): Promise<void>;
  getQuickEntryShortcut(): Promise<string | null>;
  registerQuickEntryShortcut(accelerator: string): Promise<boolean>;
  unregisterQuickEntryShortcut(): Promise<void>;
  setWindowTheme(dark: boolean): Promise<boolean>;
};
