import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SavedChats from "./saved-chats";

const { chatStore, navigate } = vi.hoisted(() => ({
  chatStore: {
    savedChats: [{ id: "saved-chat", title: "Saved chat", createdAt: 0, updatedAt: 0, messageCount: 2 }],
    loadSavedChats: vi.fn(),
    deleteChat: vi.fn(async () => {}),
    updateChatTitle: vi.fn(),
  },
  navigate: vi.fn(),
}));

vi.mock("@/stores/use-chat-store", () => ({
  useChatStore: (selector: (state: typeof chatStore) => unknown) => selector(chatStore),
}));

vi.mock("@tanstack/react-router", () => ({
  useNavigate: () => navigate,
  useParams: () => ({ chatId: "another-chat" }),
}));

vi.mock("./chat-delete-modal", () => ({
  ChatDeleteModal: ({ isOpen, onConfirm }: { isOpen: boolean; onConfirm: () => void }) =>
    isOpen ? (
      <div role="dialog" aria-label="Delete chat">
        <button onClick={onConfirm}>Confirm deletion</button>
      </div>
    ) : null,
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
});

afterEach(() => {
  cleanup();
  fireEvent.keyUp(window, { key: "Shift" });
  vi.unstubAllGlobals();
});

async function openMenu() {
  render(<SavedChats />);
  fireEvent.click(screen.getByRole("button", { name: "Options for Saved chat" }));
  return screen.findByRole("menu");
}

it("keeps the open menu mounted while Shift is pressed and released", async () => {
  const menu = await openMenu();

  fireEvent.keyDown(menu, { key: "Shift", shiftKey: true });
  expect(screen.getByRole("menu")).toBe(menu);
  expect(screen.getByRole("menuitem", { name: "Delete" })).toBeDefined();

  fireEvent.keyUp(menu, { key: "Shift" });
  expect(screen.getByRole("menu")).toBe(menu);
  expect(chatStore.deleteChat).not.toHaveBeenCalled();
  expect(navigate).not.toHaveBeenCalled();
});

it("deletes without confirmation when Shift-clicking Delete in an open menu", async () => {
  const menu = await openMenu();
  fireEvent.keyDown(menu, { key: "Shift", shiftKey: true });
  fireEvent.click(screen.getByRole("menuitem", { name: "Delete" }), { shiftKey: true });

  await waitFor(() => expect(chatStore.deleteChat).toHaveBeenCalledWith("saved-chat"));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(navigate).not.toHaveBeenCalled();
});

it("deletes without confirmation when activating Delete with Shift+Enter", async () => {
  const menu = await openMenu();
  fireEvent.keyDown(menu, { key: "End" });
  const deleteItem = screen.getByRole("menuitem", { name: "Delete" });
  await waitFor(() => expect(menu.getAttribute("aria-activedescendant")).toBe(deleteItem.id));

  fireEvent.keyDown(menu, { key: "Shift", shiftKey: true });
  fireEvent.keyDown(menu, { key: "Enter", shiftKey: true });

  await waitFor(() => expect(chatStore.deleteChat).toHaveBeenCalledWith("saved-chat"));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(navigate).not.toHaveBeenCalled();
});

it("asks for confirmation when deleting after releasing Shift", async () => {
  const menu = await openMenu();
  fireEvent.keyDown(menu, { key: "Shift", shiftKey: true });
  fireEvent.keyUp(menu, { key: "Shift" });
  fireEvent.click(screen.getByRole("menuitem", { name: "Delete" }));

  expect(screen.getByRole("dialog", { name: "Delete chat" })).toBeDefined();
  expect(chatStore.deleteChat).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Confirm deletion" }));
  await waitFor(() => expect(chatStore.deleteChat).toHaveBeenCalledWith("saved-chat"));
});

it("keeps Shift quick-delete available while the menu is closed", async () => {
  render(<SavedChats />);
  fireEvent.keyDown(window, { key: "Shift", shiftKey: true });
  fireEvent.click(screen.getByRole("button", { name: "Delete Saved chat" }), { shiftKey: true });

  await waitFor(() => expect(chatStore.deleteChat).toHaveBeenCalledWith("saved-chat"));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.queryByRole("menu")).toBeNull();
  expect(navigate).not.toHaveBeenCalled();
});
