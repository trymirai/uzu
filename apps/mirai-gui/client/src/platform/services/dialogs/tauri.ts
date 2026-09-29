import { invoke } from "../shared/invoke";
import { save } from "@tauri-apps/plugin-dialog";
import type { DialogsService } from ".";

export const tauriDialogs: DialogsService = {
  showSaveDialog: async (options) => {
    try {
      const path = await save({
        title: options.title,
        defaultPath: options.defaultPath,
        filters: options.filters,
      });
      return path ?? null;
    } catch {
      return null;
    }
  },
  readTextFile: async (absolutePath) => {
    try {
      return await invoke<string | null>("read_text_file", { absolutePath });
    } catch {
      return null;
    }
  },
};
