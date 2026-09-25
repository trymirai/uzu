import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { ChatData } from "@/platform/services/storage";
import { runLlmStream, type RunEvent, type RunTransport } from "@/platform/services/chat/runStream";
import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import { Roles } from "@/types/chat";
import { useActiveAssistantBuffer } from "./useActiveAssistantBuffer";
import { useChatSwitchEffect } from "./useChatSwitchEffect";
import { useLlmStream, type StartStreamOptions } from "./useLlmStream";

const mocks = vi.hoisted(() => ({ chats: new Map<string, ChatData>() }));
vi.mock("@/platform/platformSingleton", () => ({
  getPlatform: () => ({
    storage: {
      loadChat: async (id: string) => mocks.chats.get(id) ?? null,
      listChats: async () => [...mocks.chats.values()].map((c) => c.metadata),
    },
  }),
}));

const CHAT_A = "a";
const CHAT_B = "b";
const MESSAGE_ID = "assistant-1";
const sessionDefaults = useChatSessionStore.getState();
const chatDefaults = useChatStore.getState();

const chatData = (id: string, messages: ChatData["messages"]): ChatData => ({
  metadata: { id, title: "Untitled", messageCount: messages.length, createdAt: 1, updatedAt: 1 },
  messages,
});

const usePageWiring = (chatId: string) => {
  const stream = useLlmStream(chatId);
  const replay = useActiveAssistantBuffer(chatId);
  useChatSwitchEffect({ chatId, isNewChat: false, replayActiveBuffer: replay, resetLocalUi: () => {} });
  return stream;
};

beforeEach(() => {
  // A fixed 100 ms per frame keeps the reveal pacing off the wall clock.
  let frameNow = 0;
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) =>
    window.setTimeout(() => cb((frameNow += 100)), 0),
  );
  vi.stubGlobal("cancelAnimationFrame", (id: number) => window.clearTimeout(id));
  useChatSessionStore.setState(sessionDefaults, true);
  useChatStore.setState(chatDefaults, true);
  mocks.chats.clear();
  mocks.chats.set(
    CHAT_A,
    chatData(CHAT_A, [
      { id: "user-1", text: "hi", sender: Roles.User, timestamp: 1 },
      { id: MESSAGE_ID, text: "", sender: Roles.Assistant, timestamp: 2 },
    ]),
  );
  mocks.chats.set(CHAT_B, chatData(CHAT_B, [{ id: "user-b", text: "other", sender: Roles.User, timestamp: 3 }]));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

it("keeps streaming into the message after leaving and reopening the chat", async () => {
  let emit: (event: RunEvent) => void = () => {};
  const transport: RunTransport = {
    start: (_runId, _params, onEvent) => {
      emit = onEvent;
      return new Promise(() => {});
    },
    cancel: vi.fn(() => Promise.resolve()),
  };
  useChatStore.setState({ runChatStream: (params) => runLlmStream(transport, params) });

  const page = renderHook(({ chatId }) => usePageWiring(chatId), { initialProps: { chatId: CHAT_A } });
  await waitFor(() => expect(useChatStore.getState().messages.map((m) => m.id)).toEqual(["user-1", MESSAGE_ID]));

  const options: StartStreamOptions = {
    repoId: "vendor/model",
    chatId: CHAT_A,
    messageId: MESSAGE_ID,
    messages: [{ role: "user", content: "hi" }],
    updateText: (id, text) => useChatStore.getState().updateMessage(id, { text }),
    onError: vi.fn(),
  };
  act(() => {
    void page.result.current.startStream(options);
  });
  await waitFor(() => expect(page.result.current.isStreaming).toBe(true));
  act(() => emit({ type: "chunk", delta: "Hello" }));
  await waitFor(() => expect(useChatStore.getState().messages.at(-1)?.text).toBe("Hello"));

  page.rerender({ chatId: CHAT_B });
  await waitFor(() => expect(useChatStore.getState().currentChatId).toBe(CHAT_B));
  act(() => emit({ type: "chunk", delta: " world" }));

  page.rerender({ chatId: CHAT_A });
  await waitFor(() => expect(useChatStore.getState().messages.map((m) => m.id)).toEqual(["user-1", MESSAGE_ID]));
  await waitFor(() => expect(useChatStore.getState().messages.at(-1)?.text).toBe("Hello world"));

  act(() => emit({ type: "chunk", delta: "!" }));
  await waitFor(() => expect(useChatStore.getState().messages.at(-1)?.text).toBe("Hello world!"));
});
