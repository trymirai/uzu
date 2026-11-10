import { browserFiles } from "../shared/browser-db";
import type { ChatFiles } from "./chat-repository";

const PREFIX = "chats/";

export const browserChatFiles: ChatFiles = {
  // Another tab can write the same chat; the in-tab file queue cannot see it.
  withLock: async (key, run) => (navigator.locks ? await navigator.locks.request(`mirai.chat.${key}`, run) : run()),
  listChatFiles: async () =>
    (await browserFiles.keys()).filter((key) => key.startsWith(PREFIX)).map((key) => key.slice(PREFIX.length)),
  loadChatFile: async (path) => (await browserFiles.get(`${PREFIX}${path}`)) ?? null,
  saveChatFile: (path, content) => browserFiles.put(`${PREFIX}${path}`, content),
  deleteChatFile: (path) => browserFiles.delete(`${PREFIX}${path}`),
};
