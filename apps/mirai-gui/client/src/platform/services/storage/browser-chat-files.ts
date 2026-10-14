import { browserDb } from "../shared/browser-db";
import type { ChatFiles } from "./chat-repository";

const PREFIX = "chats/";

export const browserChatFiles: ChatFiles = {
  withLock: async (key, run) => (navigator.locks ? await navigator.locks.request(`mirai.chat.${key}`, run) : run()),
  listChatFiles: async () =>
    (await browserDb.keys("files")).filter((key) => key.startsWith(PREFIX)).map((key) => key.slice(PREFIX.length)),
  loadChatFile: async (path) => (await browserDb.get<string>("files", `${PREFIX}${path}`)) ?? null,
  saveChatFile: (path, content) => browserDb.put("files", `${PREFIX}${path}`, content),
  deleteChatFile: (path) => browserDb.delete("files", `${PREFIX}${path}`),
};
