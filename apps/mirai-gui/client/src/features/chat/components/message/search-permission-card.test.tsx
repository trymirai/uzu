import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useWebSearchStore } from "@/stores/use-web-search-store";
import type { SearchPermissionDecision } from "@/types/web-search";
import { SearchPermissionCard } from "./search-permission-card";

const platform = vi.hoisted(() => ({
  chat: {
    answerSearchPermission: vi
      .fn<(requestId: string, shownItems: number, decision: SearchPermissionDecision) => Promise<boolean>>()
      .mockResolvedValue(true),
  },
  settings: { setWebSearchModes: vi.fn(async () => undefined) },
}));

vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => platform }));

const request = { requestId: "run:1", provider: "DuckDuckGo", queries: ["rust stable version"], pages: [] };
const shownIn = { chatId: "chat", messageId: "message" };

function Card() {
  const pending = useChatSessionStore((s) => s.searchPermission);
  return pending && <SearchPermissionCard request={pending} />;
}

const show = (queries: string[], pages: string[] = []) => {
  useChatSessionStore.getState().setSearchPermission({ ...request, queries, pages, ...shownIn });
  render(<Card />);
};

const chatMode = () => useWebSearchStore.getState().chats.chat;

// Queries appear whole instead of being typed, so the answers are there at once.
beforeAll(() => {
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: query.includes("reduce"), media: query }));
});

beforeEach(() => {
  vi.clearAllMocks();
  useChatSessionStore.setState({ searchPermission: null });
  useWebSearchStore.setState({ chats: { chat: "ask" }, last: "ask" });
});
afterEach(cleanup);

it("asks about an address the user wrote without offering to always allow", () => {
  const address = "https://docs.example/d/1?key=secret";
  show([], [address]);
  expect(screen.getByRole("group", { name: "Read this page with DuckDuckGo?" })).toBeTruthy();
  expect(screen.getByRole("group", { name: "Read this page with DuckDuckGo?" }).textContent).toContain(
    address.replace("https://", ""),
  );
  expect(screen.queryByRole("button", { name: "More ways to allow" })).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Allow" }));
  expect(platform.chat.answerSearchPermission).toHaveBeenCalledExactlyOnceWith("run:1", 1, "allow");
});

it.each([
  ["Deny", "deny"],
  ["Allow", "allow"],
] as const)("%s answers the request as it was shown and leaves the chat's mode alone", async (label, decision) => {
  show(request.queries);
  fireEvent.click(screen.getByRole("button", { name: label }));
  expect(platform.chat.answerSearchPermission).toHaveBeenCalledExactlyOnceWith("run:1", 1, decision);
  await waitFor(() => expect(useChatSessionStore.getState().searchPermission).toBeNull());
  expect(chatMode()).toBe("ask");
});

it("Always allow sets the chat to search without asking once the backend has taken the answer", async () => {
  show(request.queries);
  fireEvent.click(screen.getByRole("button", { name: "More ways to allow" }));
  fireEvent.click(await screen.findByRole("menuitem", { name: /Always allow/ }));
  expect(platform.chat.answerSearchPermission).toHaveBeenCalledExactlyOnceWith("run:1", 1, "alwaysAllow");
  expect(chatMode()).toBe("ask");
  await waitFor(() => expect(chatMode()).toBe("auto"));
  expect(useWebSearchStore.getState()).toMatchObject({ chats: { chat: "auto" }, last: "ask" });
});

it.each([
  ["refused", () => Promise.resolve(false)],
  ["failed", () => Promise.reject(new Error("ipc"))],
] as const)("keeps the card, usable again, when the answer is %s", async (_outcome, respond) => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  let finish!: () => void;
  const sent = new Promise<void>((resolve) => (finish = resolve));
  platform.chat.answerSearchPermission.mockImplementationOnce(() => sent.then(respond));
  show(request.queries);
  const allow = screen.getByRole("button", { name: "Allow" });

  fireEvent.click(allow);
  expect(allow).toHaveProperty("disabled", true);
  fireEvent.click(screen.getByRole("button", { name: "Deny" }));
  expect(platform.chat.answerSearchPermission).toHaveBeenCalledTimes(1);

  const joined = { ...request, queries: ["rust stable version", "second query"] };
  useChatSessionStore.getState().setSearchPermission({ ...joined, ...shownIn });
  finish();
  await waitFor(() => expect(screen.getByRole("button", { name: "Allow" })).toHaveProperty("disabled", false));
  expect(screen.getByRole("group", { name: "Run 2 searches with DuckDuckGo?" })).toBeTruthy();
  expect(chatMode()).toBe("ask");

  fireEvent.click(screen.getByRole("button", { name: "Allow" }));
  expect(platform.chat.answerSearchPermission).toHaveBeenLastCalledWith("run:1", 2, "allow");
});
