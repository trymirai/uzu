import { beforeEach, expect, it, vi } from "vitest";
import type { ChatData } from "@/platform/services/storage";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { Roles } from "@/types/chat";
import { INTERRUPTED_REPLY_ERROR, saveInterruptedReply } from "./interrupted-reply";

type StoredMessage = ChatData["messages"][number];

const mocks = vi.hoisted(() => ({ chats: new Map<string, ChatData>() }));
vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({
    storage: {
      loadChat: async (id: string) => structuredClone(mocks.chats.get(id) ?? null),
      updateStoredMessage: async (id: string, messageId: string, patch: Partial<StoredMessage>) => {
        const chat = mocks.chats.get(id)!;
        chat.messages = chat.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m));
      },
    },
  }),
}));

const chatDefaults = useChatStore.getState();
const sessionDefaults = useChatSessionStore.getState();
const question: StoredMessage = { id: "question", sender: Roles.User, text: "question", timestamp: 1 };
const reply: StoredMessage = { id: "reply", sender: Roles.Assistant, text: "", timestamp: 2 };
const storedReply = () => mocks.chats.get("chat")?.messages[1];

beforeEach(() => {
  useChatStore.setState(chatDefaults, true);
  useChatSessionStore.setState(sessionDefaults, true);
  mocks.chats.set("chat", {
    metadata: { id: "chat", title: "Untitled", messageCount: 2, createdAt: 1, updatedAt: 1 },
    messages: [question, reply],
  });
});

it("saves the shown part of a reply in the open chat, marked interrupted", async () => {
  useChatStore.setState({ currentChatId: "chat", messages: [question, { ...reply, text: "Partial answer" }] });
  useChatSessionStore.getState().setLoadingMessage("chat", reply.id);

  await saveInterruptedReply();

  expect(storedReply()).toMatchObject({ text: "Partial answer", error: INTERRUPTED_REPLY_ERROR });
});

it("saves a reply streaming into a chat that is not open from the stream buffer", async () => {
  const session = useChatSessionStore.getState();
  session.setLoadingMessage("chat", reply.id);
  session.setActiveGenerating("chat", reply.id);
  session.setActiveAssistantMessageText("Buffered answer");

  await saveInterruptedReply();

  expect(storedReply()).toMatchObject({ text: "Buffered answer", error: INTERRUPTED_REPLY_ERROR });
});

it("changes nothing when no reply is generating", async () => {
  await saveInterruptedReply();

  expect(storedReply()).toEqual(reply);
});
