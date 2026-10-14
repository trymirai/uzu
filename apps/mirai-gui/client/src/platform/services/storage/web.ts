import type { StorageService } from ".";
import { createChatRepository } from "./chat-repository";
import { browserChatFiles } from "./browser-chat-files";
import { buildChatsZip } from "./export-zip";
import { browserDb } from "../shared/browser-db";
import { withFileQueue } from "./file-queue";
import { webModels } from "../models/web";
import type { EngineModel } from "@/types/model-manager";

const isSkipped = (model: EngineModel, identifiers: string[]): boolean =>
  identifiers.includes(model.identifier) || (model.repoId != null && identifiers.includes(model.repoId));

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
    withFileQueue("global-instructions", () => browserDb.put("files", "global-instructions", content)),
  loadGlobalInstructions: async () => (await browserDb.get<string>("files", "global-instructions")) ?? null,
  async previewCleanup(skipModelIdentifiers = []) {
    const files = await browserChatFiles.listChatFiles();
    let chatBytes = 0;
    for (const file of files)
      chatBytes += new TextEncoder().encode((await browserChatFiles.loadChatFile(file)) ?? "").byteLength;
    const { models } = await webModels.getModels();
    const removable = models.filter(
      (model) => !isSkipped(model, skipModelIdentifiers) && model.state.downloadedKbytes > 0,
    );
    return {
      dialogs: { count: files.length, sizeBytes: chatBytes },
      models: {
        count: removable.length,
        sizeBytes: removable.reduce((sum, model) => sum + model.state.downloadedKbytes * 1024, 0),
      },
      logs: { sizeBytes: 0 },
    };
  },
  async executeCleanup(categories, skipModelIdentifiers = []) {
    const executed: string[] = [];
    const modelsSkipped: string[] = [];
    if (categories.includes("dialogs")) {
      for (const file of await browserChatFiles.listChatFiles()) {
        const chatId = file.replace(/\.md$/, "");
        await webStorage.deleteChat(chatId);
      }
      executed.push("dialogs");
    }
    if (categories.includes("models")) {
      for (const model of (await webModels.getModels()).models) {
        if (isSkipped(model, skipModelIdentifiers)) modelsSkipped.push(model.identifier);
        else await webModels.deleteModel(model.identifier);
      }
      executed.push("models");
    }
    return { executed, modelsSkipped };
  },
};
