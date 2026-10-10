import { beforeEach, expect, it, vi } from "vitest";
import { ChatNotFoundError, type ChatData, type ChatMetadata } from "@/platform/services/storage";
import { serializeToMarkdown } from "@/platform/services/storage/markdown/serialize";
import { extractMessageBlocks, parseMessage } from "@/platform/services/storage/markdown/parse";
import { useChatStore } from "@/stores/use-chat-store";
import { Roles } from "@/types/chat";

const mocks = vi.hoisted(() => ({ invoke: vi.fn(), storage: {} as Record<string, unknown> }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: mocks.invoke }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ storage: mocks.storage }) }));

import { tauriStorage } from "@/platform/services/storage/tauri";

const CHAT_A = "a";
const CHAT_B = "b";
const chatDefaults = useChatStore.getState();
const metadata: ChatMetadata = {
  id: CHAT_A,
  title: "Untitled",
  messageCount: 0,
  createdAt: 1,
  updatedAt: 1,
  modelId: "model",
  modelName: "Model",
};

const inMemoryStorage = () => {
  const chats = new Map<string, ChatData>();
  return {
    chats,
    createOrReplaceChat: vi.fn(async (chat: ChatData) => {
      chats.set(chat.metadata.id, structuredClone(chat));
    }),
    appendMessage: vi.fn(async (id: string, message: ChatData["messages"][number]) => {
      const chat = chats.get(id);
      if (chat) chat.messages = [...chat.messages.filter((m) => m.id !== message.id), structuredClone(message)];
    }),
    loadChat: vi.fn(async (id: string) => chats.get(id) ?? null),
    listChats: vi.fn(async () => [...chats.values()].map((c) => c.metadata)),
    updateStoredMessage: vi.fn(async () => {}),
    removeMessage: vi.fn(async () => {}),
  };
};

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.setState(chatDefaults, true);
  useChatStore.getState().createNewChat(CHAT_A);
  mocks.storage = inMemoryStorage();
});

it("keeps the known title when recovering a missing chat file", async () => {
  const storage = mocks.storage as ReturnType<typeof inMemoryStorage>;
  const title = "Chosen title";
  useChatStore.setState({ savedChats: [{ ...metadata, title }] });
  storage.appendMessage.mockRejectedValueOnce(new ChatNotFoundError(CHAT_A));
  const id = useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.User, text: "Question" }).id;

  await useChatStore.getState().persistMessage(CHAT_A, id);

  expect(storage.chats.get(CHAT_A)?.metadata.title).toBe(title);
  expect(useChatStore.getState().savedChats[0]?.title).toBe(title);
});

it("does not recreate an existing chat when reading it fails", async () => {
  mocks.storage = tauriStorage as unknown as Record<string, unknown>;
  mocks.invoke.mockImplementation(async (cmd: string) => {
    if (cmd === "chat_load_file") throw new Error("read failed: EIO");
    if (cmd === "chat_list_files") return [];
    return undefined;
  });
  useChatStore.setState({ savedChats: [metadata] });
  const id = useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.User, text: "new message" }).id;

  await expect(useChatStore.getState().persistMessage(CHAT_A, id)).rejects.toThrow("read failed");
  expect(mocks.invoke.mock.calls.filter(([cmd]) => cmd === "chat_save_file")).toHaveLength(0);
});

it("persists a generation error for a chat that is no longer open", async () => {
  const storage = mocks.storage as ReturnType<typeof inMemoryStorage>;
  const id = useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.Assistant, text: "" }).id;
  storage.chats.set(CHAT_A, { metadata, messages: useChatStore.getState().messages });
  useChatStore.getState().createNewChat(CHAT_B);

  await useChatStore.getState().persistMessageError(CHAT_A, id, "", "Error: Context overflow");

  expect(storage.updateStoredMessage).toHaveBeenCalledWith(
    CHAT_A,
    id,
    expect.objectContaining({ error: "Error: Context overflow" }),
  );
});

it("counts a failed final save", async () => {
  const storage = mocks.storage as ReturnType<typeof inMemoryStorage>;
  const id = useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.Assistant, text: "complete answer" }).id;
  storage.updateStoredMessage.mockRejectedValue(new Error("disk full"));

  await useChatStore.getState().finalizeAssistantMessage(CHAT_A, id, "complete answer");

  expect(useChatStore.getState().saveFailureCount).toBe(1);
  expect(useChatStore.getState().messages.find((m) => m.id === id)?.text).toBe("complete answer");
});

it("keeps one copy of a message the file already holds when appending it", async () => {
  mocks.storage = tauriStorage as unknown as Record<string, unknown>;
  const message = { id: "m1", sender: Roles.Assistant, text: "", timestamp: 1 };
  const onDisk = serializeToMarkdown({ metadata: { ...metadata, messageCount: 1 }, messages: [message] });
  let saved = "";
  mocks.invoke.mockImplementation(async (cmd: string, args: { content?: string }) => {
    if (cmd === "chat_load_file") return onDisk;
    if (cmd === "chat_save_file") saved = args.content ?? "";
    return undefined;
  });

  await tauriStorage.appendMessage(CHAT_A, { ...message, text: "done" });

  const messages = extractMessageBlocks(saved).map(parseMessage);
  expect(messages.map((m) => m?.id)).toEqual(["m1"]);
  expect(messages[0]?.text).toBe("done");
});
