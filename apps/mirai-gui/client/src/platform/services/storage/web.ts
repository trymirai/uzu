import type { StorageService } from ".";
import { createChatRepository } from "./chat-repository";
import { browserChatFiles } from "./browser-chat-files";
import { buildChatsZip } from "./export-zip";
import { withFileQueue } from "./file-queue";
import { browserFiles } from "../shared/browser-db";
import { getPlatform } from "../../platform-singleton";
import { modelDownloadPhases, type EngineModel, type ModelDownloadPhase } from "@/types/model-manager";

const GLOBAL_INSTRUCTIONS = "global-instructions";

const isSkipped = (model: EngineModel, identifiers: string[]): boolean =>
  identifiers.includes(model.identifier) || (model.repoId != null && identifiers.includes(model.repoId));

const ACTIVE_PHASES: ModelDownloadPhase[] = [
  modelDownloadPhases.initializing,
  modelDownloadPhases.downloading,
  modelDownloadPhases.paused,
  modelDownloadPhases.locked,
];
const REMOVABLE_PHASES: ModelDownloadPhase[] = [modelDownloadPhases.downloaded, modelDownloadPhases.error];

const catalogModels = async () => (await getPlatform().models.getModels()).models;

// Chats are stored as the same Markdown files as on desktop, in IndexedDB.
export const webStorage: StorageService = {
  ...createChatRepository(browserChatFiles),
  exportAllChatsZip: () => buildChatsZip(browserChatFiles),

  async saveBinaryFile(fileName, data) {
    const url = URL.createObjectURL(new Blob([new Uint8Array(data)]));
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName.split(/[\\/]/).pop() || "download";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    return true;
  },
  saveGlobalInstructions: (content) =>
    withFileQueue(GLOBAL_INSTRUCTIONS, () => browserFiles.put(GLOBAL_INSTRUCTIONS, content)),
  loadGlobalInstructions: async () => (await browserFiles.get(GLOBAL_INSTRUCTIONS)) ?? null,

  async previewCleanup(skipModelIdentifiers = []) {
    const files = await browserChatFiles.listChatFiles();
    let chatBytes = 0;
    for (const file of files)
      chatBytes += new TextEncoder().encode((await browserChatFiles.loadChatFile(file)) ?? "").byteLength;
    const downloaded = (await catalogModels()).filter(
      (model) => model.state.phase === modelDownloadPhases.downloaded && !isSkipped(model, skipModelIdentifiers),
    );
    return {
      dialogs: { count: files.length, sizeBytes: chatBytes },
      models: {
        count: downloaded.length,
        sizeBytes: downloaded.reduce((sum, model) => sum + model.state.totalKbytes * 1024, 0),
      },
      logs: { sizeBytes: 0 },
    };
  },
  async executeCleanup(categories, skipModelIdentifiers = []) {
    const executed: string[] = [];
    const modelsSkipped: string[] = [];
    if (categories.includes("dialogs")) {
      for (const file of await browserChatFiles.listChatFiles()) {
        if (file.endsWith(".md")) await webStorage.deleteChat(file.slice(0, -".md".length));
      }
      executed.push("dialogs");
    }
    if (categories.includes("models")) {
      // Like desktop: downloads in progress are skipped, and a failed deletion
      // leaves the category unexecuted so the dialog reports it.
      let allDeleted = true;
      for (const model of await catalogModels()) {
        if (ACTIVE_PHASES.includes(model.state.phase) || isSkipped(model, skipModelIdentifiers)) {
          modelsSkipped.push(model.identifier);
        } else if (REMOVABLE_PHASES.includes(model.state.phase)) {
          try {
            await getPlatform().models.deleteModel(model.repoId ?? model.identifier);
          } catch (error) {
            console.error("[cleanup] model deletion failed", { identifier: model.identifier }, error);
            allDeleted = false;
          }
        }
      }
      if (allDeleted) executed.push("models");
    }
    return { executed, modelsSkipped };
  },
};
