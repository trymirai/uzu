import type { Message } from "@/types/message";
import { ChatNotFoundError } from ".";
import type { ChatData, ChatMetadata, StorageService } from ".";
import { deleteChatFile, listChatFiles, loadChatFile, saveChatFile } from "./chat-files";
import { extractMessageBlocks, parseMessage, parseMetadata } from "./markdown/parse";
import { serializeToMarkdown } from "./markdown/serialize";
import { withFileQueue } from "./file-queue";

type ChatRepository = Pick<
  StorageService,
  | "listChats"
  | "loadChat"
  | "createOrReplaceChat"
  | "removeMessage"
  | "appendMessage"
  | "updateStoredMessage"
  | "editUserMessage"
  | "updateChatTitle"
  | "deleteChat"
>;

const writeChat = async (chat: ChatData): Promise<void> => {
  await saveChatFile(`${chat.metadata.id}.md`, serializeToMarkdown(chat));
};

const withChatLock = <T>(chatId: string, run: () => Promise<T>): Promise<T> => withFileQueue(`chat:${chatId}`, run);

// A read error must throw: appendMessage takes null for a missing chat and rewrites it from memory.
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
      // A recovery rewrite from memory may already hold this message.
      const others = existing.messages.filter((m) => m.id !== message.id);
      await writeChat({
        metadata: { ...existing.metadata, messageCount: others.length + 1, updatedAt: Date.now() },
        messages: [...others, message],
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

  editUserMessage(chatId, messageId, text) {
    return withChatLock(chatId, async () => {
      const existing = await loadChat(chatId);
      if (!existing) throw new ChatNotFoundError(chatId);
      const index = existing.messages.findIndex((message) => message.id === messageId);
      const original = existing.messages[index];
      if (!original) throw new Error(`Message ${messageId} not found in chat ${chatId}`);
      if (original.sender !== "user") throw new Error("Only user messages can be edited");
      const edited: Message = {
        ...original,
        text,
        // Loaded user messages also have parsed output, which the serializer
        // prefers over text. Keep that projection in sync with the edit.
        ...(original.output
          ? {
              output: {
                ...original.output,
                text: {
                  ...original.output.text,
                  parsed: { ...original.output.text?.parsed, response: text },
                },
              },
            }
          : {}),
      };
      const messages = [...existing.messages.slice(0, index), edited];
      const saved = {
        metadata: { ...existing.metadata, messageCount: messages.length, updatedAt: Date.now() },
        messages,
      };
      await writeChat(saved);
      return saved;
    });
  },

  removeMessage(chatId, messageId) {
    return withChatLock(chatId, async () => {
      const existing = await loadChat(chatId);
      if (!existing) return;
      const messages = existing.messages.filter((m) => m.id !== messageId);
      if (messages.length === existing.messages.length) return;
      await writeChat({ metadata: { ...existing.metadata, messageCount: messages.length }, messages });
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
