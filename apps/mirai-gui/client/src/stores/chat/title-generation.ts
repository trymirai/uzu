import { getPlatform } from "@/platform/platform-singleton";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { CHAT_TITLE_MAX_LENGTH, DEFAULT_CHAT_TITLE, UNTITLED_CHAT_TITLE } from "@/constants/chat";
import { Roles } from "@/types/chat";
import type { ChatStoreApi } from "./types";

const sanitizeChatTitle = (raw: string): string => {
  if (!raw) return "";
  let title = raw.trim();
  title = title.replace(/[\r\n]+/g, " ");
  title = title.replace(/\s+/g, " ").trim();
  title = title.replace(/^["'`]+|["'`]+$/g, "");
  return title;
};

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
  const existingTitle = chatData?.metadata.title ?? UNTITLED_CHAT_TITLE;
  if (!isPlaceholderChatTitle(existingTitle)) return;

  const repoIdToUse = chatModel?.modelId || chatData?.metadata.modelId;
  if (!repoIdToUse) return;

  const firstUserQuoted = userText.replace(/\s+/g, " ").trim().replace(/"/g, '\\"');
  // Small models copy a fallback phrase from the prompt instead of titling.
  const userOnlyInstruction =
    `Write a chat title of 2 to 4 words for the message below. Reply with the title only, no quotes, no punctuation at the end. ` +
    `Message: "${firstUserQuoted}"`;

  const title = await getPlatform().chat.generateTitle({
    repoId: repoIdToUse,
    messages: [{ role: Roles.User, content: userOnlyInstruction }],
  });
  // A cancelled generation returns whatever came out before the cancel, usually
  // nothing; writing that or the default here would stop the next message from
  // generating a real title.
  if (useChatSessionStore.getState().titleGenAbortChatId === chatId) return;

  const candidateRaw = sanitizeChatTitle(title);
  // Longer than a manual rename allows means the model answered instead of titling.
  const candidate = candidateRaw.length > CHAT_TITLE_MAX_LENGTH ? "New chat" : candidateRaw || DEFAULT_CHAT_TITLE;
  const { storage } = getPlatform();
  // The user may have renamed the chat while the model was thinking.
  await storage.updateChatTitle(chatId, candidate, existingTitle);
  const savedChats = await storage.listChats();
  set({ savedChats });
};
