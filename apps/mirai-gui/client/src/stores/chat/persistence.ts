import { getPlatform } from "@/platform/platform-singleton";
import { ChatNotFoundError } from "@/platform/services/storage";
import type { ChatMetadata } from "@/platform/services/storage";
import { UNTITLED_CHAT_TITLE } from "@/constants/chat";
import type { Message, ParsedOutput } from "@/types/message";
import type { TranscriptItem } from "@/types/llm-stream";
import { computeErrorPatch, computeFinalizedUpdates } from "./message-patches";
import type { ChatStoreApi } from "./types";

const reportSaveFailure = (set: ChatStoreApi["set"]) => set((s) => ({ saveFailureCount: s.saveFailureCount + 1 }));

export const persistMessage = async (
  { get, set }: ChatStoreApi,
  chatId: string,
  messageId: string,
  known?: Message,
): Promise<void> => {
  const state = get();
  const message = known ?? state.messages.find((m) => m.id === messageId);
  if (!message) return;
  const { storage } = getPlatform();
  const chatExists = state.savedChats.some((c) => c.id === chatId);
  if (chatExists) {
    try {
      await storage.appendMessage(chatId, message);
    } catch (e) {
      if (!(e instanceof ChatNotFoundError)) throw e;
      const chatModel = get().chatModels[chatId];
      const now = Date.now();
      const fallbackMetadata: ChatMetadata = {
        id: chatId,
        title: UNTITLED_CHAT_TITLE,
        modelId: chatModel?.modelId,
        modelName: chatModel?.modelName,
        createdAt: now,
        updatedAt: now,
        messageCount: get().messages.length,
      };
      set((s) => ({ savedChats: [fallbackMetadata, ...s.savedChats.filter((c) => c.id !== chatId)] }));
      // Don't merge with foreign-chat messages if user navigated away mid-recovery.
      const currentMessages = get().currentChatId === chatId ? get().messages : [message];
      await storage.createOrReplaceChat({
        metadata: { ...fallbackMetadata, messageCount: currentMessages.length },
        messages: currentMessages,
      });
    }
  } else {
    const chatModel = state.chatModels[chatId];
    const now = Date.now();
    // savedChats loads lazily: this chat may already have a file, which a partial write would truncate.
    const messages = state.currentChatId === chatId ? state.messages : [message];
    const newMetadata: ChatMetadata = {
      id: chatId,
      title: UNTITLED_CHAT_TITLE,
      modelId: chatModel?.modelId,
      modelName: chatModel?.modelName,
      createdAt: now,
      updatedAt: now,
      messageCount: messages.length,
    };
    // Mark chat as existing before the await so a concurrent persistMessage
    // takes the append path instead of racing into a second createOrReplaceChat.
    set((s) => ({ savedChats: [newMetadata, ...s.savedChats] }));
    await storage.createOrReplaceChat({ metadata: newMetadata, messages });
  }
  const savedChats = await storage.listChats();
  set({ savedChats });
};

export const persistMessagePatch = async (
  { set }: ChatStoreApi,
  chatId: string,
  messageId: string,
  patch: Partial<Message>,
): Promise<void> => {
  try {
    await getPlatform().storage.updateStoredMessage(chatId, messageId, patch);
  } catch (e) {
    console.error("[storage] persistMessagePatch failed", { chatId, messageId }, e);
    reportSaveFailure(set);
  }
};

export const persistMessageError = async (
  { get, set }: ChatStoreApi,
  chatId: string,
  messageId: string,
  text: string,
  error: string,
  attachmentIds?: string[],
): Promise<void> => {
  const { storage } = getPlatform();
  try {
    const message = await findMessage({ get }, storage, chatId, messageId);
    if (!message) return;
    const patch = computeErrorPatch(message, text, error, attachmentIds);
    await storage.updateStoredMessage(chatId, messageId, patch);
  } catch (e) {
    console.error("[storage] persistMessageError failed", { chatId, messageId }, e);
    reportSaveFailure(set);
  }
};

// The user may have opened another chat while this one was generating; then
// the message is only on disk.
const findMessage = async (
  { get }: Pick<ChatStoreApi, "get">,
  storage: ReturnType<typeof getPlatform>["storage"],
  chatId: string,
  messageId: string,
): Promise<Message | undefined> => {
  if (get().currentChatId === chatId) return get().messages.find((m) => m.id === messageId);
  const chatData = await storage.loadChat(chatId);
  return chatData?.messages.find((m) => m.id === messageId);
};

export const finalizeAssistantMessage = async (
  { get, set }: ChatStoreApi,
  chatId: string,
  messageId: string,
  text: string,
  parsed?: ParsedOutput,
  transcript?: TranscriptItem[],
): Promise<void> => {
  const { storage } = getPlatform();
  const inView = get().currentChatId === chatId;
  if (inView) {
    const current = get().messages.find((m) => m.id === messageId);
    if (!current) return;
    get().updateMessage(messageId, computeFinalizedUpdates(current, text, parsed, transcript));
  }
  try {
    // Read back so perf/stats written by applyPerf right before finalize
    // make it into the persisted patch.
    const latest = await findMessage({ get }, storage, chatId, messageId);
    if (!latest) return;
    const patch: Partial<Message> = {
      ...computeFinalizedUpdates(latest, text, parsed, transcript),
      ...(inView && latest.perf ? { perf: latest.perf } : {}),
      ...(inView && latest.stats ? { stats: latest.stats } : {}),
    };
    await storage.updateStoredMessage(chatId, messageId, patch);
    set({ savedChats: await storage.listChats() });
  } catch (e) {
    console.error("[storage] finalizeAssistantMessage failed", { chatId, messageId }, e);
    reportSaveFailure(set);
  }
};
