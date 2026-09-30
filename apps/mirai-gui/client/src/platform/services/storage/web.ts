import type { StorageService } from ".";

// Inference is native, so the browser build can never produce a chat. Nothing
// is persisted here: the calls exist only to satisfy the platform contract.
export const webStorage: StorageService = {
  listChats: () => Promise.resolve([]),
  loadChat: () => Promise.resolve(null),
  createOrReplaceChat: () => Promise.resolve(),
  appendMessage: () => Promise.resolve(),
  updateStoredMessage: () => Promise.resolve(),
  removeMessage: () => Promise.resolve(),
  updateChatTitle: () => Promise.resolve(),
  deleteChat: () => Promise.resolve(),
  exportAllChatsZip: () => Promise.resolve(null),

  saveBinaryFile: () => Promise.resolve(false),
  saveGlobalInstructions: () => Promise.resolve(),
  loadGlobalInstructions: () => Promise.resolve(null),

  previewCleanup: () =>
    Promise.resolve({
      dialogs: { count: 0, sizeBytes: 0 },
      models: { count: 0, sizeBytes: 0 },
      logs: { sizeBytes: 0 },
    }),
  executeCleanup: () => Promise.resolve({ executed: [], modelsSkipped: [] }),
};
