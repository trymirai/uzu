import { invoke } from "../shared/invoke";
import type { StorageCleanupPreview, StorageService } from ".";
import { chatRepository } from "./chat-repository";
import { buildChatsZip } from "./export-zip";
import { withFileQueue } from "./file-queue";

export const tauriStorage: StorageService = {
  ...chatRepository,
  exportAllChatsZip: buildChatsZip,

  saveBinaryFile: (absolutePath, data) => invoke<boolean>("save_binary_file", { absolutePath, data: Array.from(data) }),
  saveGlobalInstructions: (content) =>
    withFileQueue("global-instructions", () => invoke<void>("global_instructions_save", { content })),
  loadGlobalInstructions: () => invoke<string | null>("global_instructions_load"),

  previewCleanup: (skipModelIdentifiers) =>
    invoke<StorageCleanupPreview>("cleanup_preview", { skipModelIdentifiers: skipModelIdentifiers ?? [] }),
  executeCleanup: (categories, skipModelIdentifiers) =>
    invoke<{ executed: string[]; modelsSkipped: string[] }>("cleanup_execute", {
      categories,
      skipModelIdentifiers: skipModelIdentifiers ?? [],
    }),
};
