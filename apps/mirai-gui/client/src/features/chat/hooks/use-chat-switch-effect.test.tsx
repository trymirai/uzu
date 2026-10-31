import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { ChatData, ChatMetadata } from "@/platform/services/storage";
import { useChatStore } from "@/stores/use-chat-store";
import { Roles } from "@/types/chat";
import { useChatSwitchEffect } from "./use-chat-switch-effect";

const mocks = vi.hoisted(() => ({ chats: new Map<string, ChatData>() }));
vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({ storage: { loadChat: async (id: string) => structuredClone(mocks.chats.get(id) ?? null) } }),
}));

const CHAT = "chat";
const chatDefaults = useChatStore.getState();
const metadata: ChatMetadata = {
  id: CHAT,
  title: "Untitled",
  messageCount: 2,
  createdAt: 1,
  updatedAt: 1,
  modelId: "model",
  modelName: "Model",
};
const messages: ChatData["messages"] = [
  { id: "question", sender: Roles.User, text: "question", timestamp: 1 },
  { id: "answer", sender: Roles.Assistant, text: "answer", timestamp: 2 },
];

const renderSwitch = (isNewChat: boolean, forgetNewChat = vi.fn()) =>
  renderHook(
    (props: { isNewChat: boolean }) =>
      useChatSwitchEffect({
        chatId: CHAT,
        isNewChat: props.isNewChat,
        replayActiveBuffer: () => {},
        resetLocalUi: () => {},
        forgetNewChat,
      }),
    { initialProps: { isNewChat } },
  );

beforeEach(() => {
  useChatStore.setState(chatDefaults, true);
  mocks.chats.clear();
});
afterEach(cleanup);

it("drops the new-chat flag once the chat is saved, keeping the messages in view", () => {
  const forgetNewChat = vi.fn();
  const view = renderSwitch(true, forgetNewChat);
  expect(forgetNewChat).not.toHaveBeenCalled();

  act(() => useChatStore.setState({ messages: messages.slice(0, 1), savedChats: [metadata] }));
  expect(forgetNewChat).toHaveBeenCalledOnce();

  view.rerender({ isNewChat: false });
  expect(useChatStore.getState().messages).toEqual(messages.slice(0, 1));
});

it("loads a saved chat and its model when the page opens it anew", async () => {
  mocks.chats.set(CHAT, { metadata, messages });

  renderSwitch(false);

  await waitFor(() => expect(useChatStore.getState().messages).toEqual(messages));
  expect(useChatStore.getState().chatModels[CHAT]).toEqual({ modelId: "model", modelName: "Model" });
});
