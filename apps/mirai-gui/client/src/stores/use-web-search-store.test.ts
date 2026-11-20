import { beforeEach, expect, it, vi } from "vitest";
import type { WebSearchModes } from "@/platform/services/settings";
import { useWebSearchStore, webSearchModeOf } from "./use-web-search-store";

const settings = vi.hoisted(() => ({
  getWebSearchModes: vi.fn(async (): Promise<WebSearchModes> => ({ chats: {}, last: "off" })),
  setWebSearchModes: vi.fn<(modes: WebSearchModes) => Promise<void>>().mockResolvedValue(undefined),
}));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings }) }));

const defaults = useWebSearchStore.getState();
const mode = (chatId: string) => webSearchModeOf(useWebSearchStore.getState(), chatId);

beforeEach(() => {
  vi.clearAllMocks();
  useWebSearchStore.setState(defaults, true);
});

it("keeps each chat's mode, starts a new chat with the latest one and saves both to the settings", () => {
  useWebSearchStore.getState().markNew("new");
  expect(mode("new")).toBe("off");

  useWebSearchStore.getState().setMode("first", "auto");
  expect(mode("first")).toBe("auto");
  expect(mode("new")).toBe("auto");
  expect(mode("older"), "a chat from before the modes were kept").toBe("off");
  expect(settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: { first: "auto" }, last: "auto" });

  useWebSearchStore.getState().setMode("second", "off");
  expect(mode("first")).toBe("auto");
  expect(mode("another")).toBe("off");
  expect(settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: { first: "auto", second: "off" }, last: "off" });
});

it("sets one chat's mode without changing what the other chats and a new chat follow", () => {
  useWebSearchStore.getState().markNew("new");
  useWebSearchStore.getState().setMode("first", "ask");

  useWebSearchStore.getState().setChatMode("second", "auto");
  expect(mode("second")).toBe("auto");
  expect(mode("first")).toBe("ask");
  expect(mode("new")).toBe("ask");
  expect(settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: { first: "ask", second: "auto" }, last: "ask" });
});

it("reads the saved modes back", async () => {
  settings.getWebSearchModes.mockResolvedValueOnce({ chats: { asked: "ask" }, last: "auto" });
  await useWebSearchStore.getState().load();
  useWebSearchStore.getState().markNew("new");
  expect(mode("asked")).toBe("ask");
  expect(mode("new")).toBe("auto");
});

it("forgets a deleted chat and leaves the others and the latest mode", () => {
  useWebSearchStore.getState().setMode("first", "ask");
  useWebSearchStore.getState().setMode("second", "auto");
  settings.setWebSearchModes.mockClear();

  useWebSearchStore.getState().forgetChat("first");
  expect(settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: { second: "auto" }, last: "auto" });
  expect(mode("first")).toBe("off");
});

it("forgets every chat and leaves the latest mode for the next one", () => {
  useWebSearchStore.getState().setMode("first", "ask");
  useWebSearchStore.getState().setMode("second", "auto");

  useWebSearchStore.getState().forgetAll();
  expect(settings.setWebSearchModes).toHaveBeenLastCalledWith({ chats: {}, last: "auto" });
  useWebSearchStore.getState().markNew("next");
  expect(mode("next")).toBe("auto");
});
