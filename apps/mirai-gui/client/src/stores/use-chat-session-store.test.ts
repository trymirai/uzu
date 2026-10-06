import { beforeEach, expect, it } from "vitest";
import { useChatSessionStore } from "./use-chat-session-store";

const defaults = useChatSessionStore.getState();

beforeEach(() => {
  useChatSessionStore.setState(defaults, true);
});

it("ignores a loading-message clear that comes from another chat", () => {
  const { setLoadingMessage } = useChatSessionStore.getState();
  setLoadingMessage("a", "m1");

  setLoadingMessage("b", null);
  expect(useChatSessionStore.getState().loadingMessage).toEqual({ chatId: "a", messageId: "m1" });

  setLoadingMessage("a", null);
  expect(useChatSessionStore.getState().loadingMessage).toBeNull();
});

it("marks only the stopped chat's message as canceled", () => {
  const { setCanceledMessage } = useChatSessionStore.getState();
  setCanceledMessage("a", "m1");

  setCanceledMessage("b", null);
  expect(useChatSessionStore.getState().canceledMessage).toEqual({ chatId: "a", messageId: "m1" });
});
