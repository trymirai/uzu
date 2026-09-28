import type { Message } from "@/types/message";
import { ChatNotFoundError } from ".";
import type { ChatData, ChatMetadata, StorageService } from ".";
import { deleteChatFile, listChatFiles, loadChatFile, saveChatFile } from "./chat-files";
import { extractMessageBlocks, parseMessage, parseMetadata } from "./markdown/parse";
import { serializeToMarkdown } from "./markdown/serialize";

type ChatRepository = Pick<
  StorageService,
  | "listChats"
  | "loadChat"
  | "createOrReplaceChat"
  | "appendMessage"
  | "updateStoredMessage"
  | "updateChatTitle"
  | "deleteChat"
>;

const writeChat = async (chat: ChatData): Promise<void> => {
  await saveChatFile(`${chat.metadata.id}.md`, serializeToMarkdown(chat));
};

// Mutations are load-modify-save over the whole file; concurrent ones on the
// same chat (e.g. appendMessage racing the title-gen update) would drop each
// other's writes, so they queue per chatId.
const chains = new Map<string, Promise<unknown>>();
const withChatLock = <T>(chatId: string, run: () => Promise<T>): Promise<T> => {
  const prev = chains.get(chatId) ?? Promise.resolve();
  const next = prev.then(run, run);
  const tail: Promise<unknown> = next
    .catch(() => undefined)
    .finally(() => {
      if (chains.get(chatId) === tail) chains.delete(chatId);
    });
  chains.set(chatId, tail);
  return next;
};

// null means the file does not exist; a read failure must reject, otherwise
// appendMessage would treat the chat as missing and recreate it from memory.
const loadChat = async (chatId: string): Promise<ChatData | null> => {
  const content = await loadChatFile(`${chatId}.md`);
  if (!content) return null;
  const messages = extractMessageBlocks(content)
    .map(parseMessage)
    .filter((m): m is Message => m !== null);
  return { metadata: parseMetadata(content, chatId, messages.length), messages };
};

export const chatRepository: ChatRepository = {
  async listChats() {
    const files = await listChatFiles();
    const results = await Promise.all(
      files
        .filter((f) => f.endsWith(".md"))
        .map(async (file) => {
          try {
            const chatId = file.replace(".md", "");
            const markdown = await loadChatFile(`${chatId}.md`);
            if (!markdown) return undefined;
            const countMatch = markdown.match(/\*\*Messages:\*\*\s*(\d+)/m);
            const count = countMatch ? Number(countMatch[1]) : extractMessageBlocks(markdown).length;
            return parseMetadata(markdown, chatId, count);
          } catch {
            return undefined;
          }
        }),
    );
    return results.filter((m): m is ChatMetadata => m !== undefined).sort((a, b) => b.updatedAt - a.updatedAt);
  },

  loadChat,

  createOrReplaceChat(chat) {
    return withChatLock(chat.metadata.id, () => writeChat(chat));
  },

  appendMessage(chatId, message) {
    return withChatLock(chatId, async () => {
      const existing = await loadChat(chatId);
      if (!existing) throw new ChatNotFoundError(chatId);
      await writeChat({
        metadata: { ...existing.metadata, messageCount: existing.messages.length + 1, updatedAt: Date.now() },
        messages: [...existing.messages, message],
      });
    });
  },

  updateStoredMessage(chatId, messageId, patch) {
    return withChatLock(chatId, async () => {
      const existing = await loadChat(chatId);
      if (!existing) throw new Error(`Chat ${chatId} not found`);
      const messages = existing.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m));
      await writeChat({
        metadata: { ...existing.metadata, updatedAt: Date.now() },
        messages,
      });
    });
  },

  updateChatTitle(chatId, title, expectedTitle) {
    return withChatLock(chatId, async () => {
      const existing = await loadChat(chatId);
      if (!existing) return;
      if (expectedTitle !== undefined && existing.metadata.title !== expectedTitle) return;
      await writeChat({
        ...existing,
        metadata: { ...existing.metadata, title, updatedAt: Date.now() },
      });
    });
  },

  deleteChat(chatId) {
    return withChatLock(chatId, () => deleteChatFile(`${chatId}.md`));
  },
};
