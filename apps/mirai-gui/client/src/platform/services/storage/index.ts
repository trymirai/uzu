import type { Message } from "@/types/message";

export type ChatMetadata = {
  id: string;
  title: string;
  modelId?: string;
  modelName?: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
};

export type ChatData = {
  metadata: ChatMetadata;
  messages: Message[];
};

export type StorageCleanupPreview = {
  dialogs: { count: number; sizeBytes: number };
  models: { count: number; sizeBytes: number };
  logs: { sizeBytes: number };
};

/** Thrown by appendMessage when the chat is gone from storage; callers recreate it. */
export class ChatNotFoundError extends Error {
  constructor(chatId: string) {
    super(`Chat ${chatId} not found`);
    this.name = "ChatNotFoundError";
  }
}

export type StorageService = {
  listChats(): Promise<ChatMetadata[]>;
  loadChat(chatId: string): Promise<ChatData | null>;
  createOrReplaceChat(chat: ChatData): Promise<void>;
  appendMessage(chatId: string, message: Message): Promise<void>;
  updateStoredMessage(chatId: string, messageId: string, patch: Partial<Message>): Promise<void>;
  /** With `expectedTitle`, writes only while the stored title is still that one. */
  updateChatTitle(chatId: string, title: string, expectedTitle?: string): Promise<void>;
  deleteChat(chatId: string): Promise<void>;
  exportAllChatsZip(): Promise<Uint8Array | null>;

  saveBinaryFile(absolutePath: string, data: Uint8Array): Promise<boolean>;
  saveGlobalInstructions(content: string): Promise<void>;
  loadGlobalInstructions(): Promise<string | null>;
  previewCleanup(skipModelIdentifiers?: string[]): Promise<StorageCleanupPreview>;
  executeCleanup(
    categories: string[],
    skipModelIdentifiers?: string[],
  ): Promise<{ executed: string[]; modelsSkipped: string[] }>;
};
