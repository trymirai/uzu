import { getPlatform } from "@/platform/platform-singleton";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useModelsStore } from "@/stores/use-models-store";
import { DEFAULT_CHAT_TITLE, Roles, UNTITLED_CHAT_TITLE } from "@/types/chat";
import type { ChatStoreApi } from "./types";

// Past this length the model answered the message instead of titling it.
const MAX_TITLE_LENGTH = 300;

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

  const localModel = useModelsStore.getState().models.find((m) => m.repoId === repoIdToUse);
  const normalizedVendor = (localModel?.vendor ?? "").trim().toLowerCase();
  // LFM models fail this reliably enough that asking is wasted latency.
  if (normalizedVendor === "liquidai") {
    const { storage } = getPlatform();
    await storage.updateChatTitle(chatId, DEFAULT_CHAT_TITLE, existingTitle);
    const savedChats = await storage.listChats();
    set({ savedChats });
    return;
  }

  const firstUserQuoted = userText.replace(/\s+/g, " ").trim().replace(/"/g, '\\"');
  const userOnlyInstruction =
    `Generate a short, neutral chat title (2–4 words, no punctuation at the end) for the following user message. ` +
    `If the topic is unclear, return "${DEFAULT_CHAT_TITLE}". Message: "${firstUserQuoted}"`;

  const title = await getPlatform().chat.generateTitle({
    repoId: repoIdToUse,
    messages: [{ role: Roles.User, content: userOnlyInstruction }],
  });
  // A cancelled generation returns whatever came out before the cancel, usually
  // nothing; writing that or the default here would stop the next message from
  // generating a real title.
  if (useChatSessionStore.getState().titleGenAbortChatId === chatId) return;

  const candidateRaw = sanitizeChatTitle(title);
  const candidate = candidateRaw.length > MAX_TITLE_LENGTH ? "New chat" : candidateRaw || DEFAULT_CHAT_TITLE;
  const { storage } = getPlatform();
  // The user may have renamed the chat while the model was thinking.
  await storage.updateChatTitle(chatId, candidate, existingTitle);
  const savedChats = await storage.listChats();
  set({ savedChats });
};
