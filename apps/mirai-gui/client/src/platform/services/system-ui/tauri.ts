import { invoke } from "../shared/invoke";
import type { SystemUiService } from ".";

export const tauriSystemUi: SystemUiService = {
  setWindowTheme: (theme) => invoke<boolean>("set_window_theme", { theme }),
};
