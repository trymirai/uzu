import { getPlatform } from "@/platform/platform-singleton";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import type { ChatStoreApi } from "./types";

const isPlaceholderChatTitle = (title?: string | null): boolean => {
  const normalized = (title ?? "").trim().toLowerCase();
  return normalized.length === 0 || normalized === "untitled" || normalized === "new chat";
};

// The store's messages belong to whichever chat is open by now, so the send
// operation passes the user text itself.
export const runChatTitleGeneration = async (
  chatId: string,
  userText: string,
  get: ChatStoreApi["get"],
  set: ChatStoreApi["set"],
): Promise<void> => {
  const chatModel = get().chatModels[chatId];
  if (!chatId || userText.trim().length === 0) return;

  const chatData = await getPlatform().storage.loadChat(chatId);
  if (useChatSessionStore.getState().titleGenAbortChatId === chatId) return;
  const existingTitle = chatData?.metadata.title;
  if (existingTitle === undefined || !isPlaceholderChatTitle(existingTitle)) return;

  const repoIdToUse = chatModel?.modelId || chatData?.metadata.modelId;
  if (!repoIdToUse) return;

  const title = await getPlatform().chat.generateTitle({
    repoId: repoIdToUse,
    userText,
  });
  // A late result after Stop must not prevent the next message from trying again.
  if (!title || useChatSessionStore.getState().titleGenAbortChatId === chatId) return;
  const { storage } = getPlatform();
  // The user may have renamed the chat while the model was thinking.
  await storage.updateChatTitle(chatId, title, existingTitle);
  const savedChats = await storage.listChats();
  set({ savedChats });
};
