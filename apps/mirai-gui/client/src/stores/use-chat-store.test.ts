import { beforeEach, expect, it, vi } from "vitest";
import { useChatStore } from "./use-chat-store";
import { useWebSearchStore } from "./use-web-search-store";

const platform = vi.hoisted(() => ({
  storage: { deleteChat: vi.fn(async () => undefined) },
  settings: { setWebSearchModes: vi.fn(async () => undefined) },
}));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => platform }));
vi.mock("./use-chat-session-store", () => ({
  useChatSessionStore: { getState: () => ({ cancelActiveRunForChat: async () => undefined }) },
}));

beforeEach(() => {
  vi.clearAllMocks();
  useWebSearchStore.setState({ chats: {}, last: "off" });
});

it("drops a deleted chat's web search mode from the settings and keeps the others", async () => {
  useWebSearchStore.getState().setMode("kept", "auto");
  useWebSearchStore.getState().setMode("deleted", "ask");

  await useChatStore.getState().deleteChat("deleted");

  expect(platform.storage.deleteChat).toHaveBeenCalledExactlyOnceWith("deleted");
  expect(useWebSearchStore.getState().chats).toEqual({ kept: "auto" });
  expect(platform.settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: { kept: "auto" }, last: "ask" });
});
