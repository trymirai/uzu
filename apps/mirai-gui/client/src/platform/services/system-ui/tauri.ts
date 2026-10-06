import { invoke } from "../shared/invoke";
import type { SystemUiService } from ".";

export const tauriSystemUi: SystemUiService = {
  getRunOnStartup: () => invoke<boolean>("get_run_on_startup"),
  setRunOnStartup: (value) => invoke<void>("set_run_on_startup", { value }),
  getQuickEntryShortcut: () => invoke<string | null>("get_quick_entry_shortcut"),
  registerQuickEntryShortcut: (accelerator) => invoke<boolean>("register_quick_entry_shortcut", { accelerator }),
  unregisterQuickEntryShortcut: () => invoke<void>("unregister_quick_entry_shortcut"),
  setWindowTheme: (dark) => invoke<boolean>("set_window_theme", { dark }),
};
