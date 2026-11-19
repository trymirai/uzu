import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCleanupStore } from "@/stores/use-cleanup-store";
import { useWebSearchStore } from "@/stores/use-web-search-store";
import { ClearDataDialog } from "./clear-data-dialog";

const platform = vi.hoisted(() => ({
  storage: {
    previewCleanup: vi.fn(async () => ({
      dialogs: { count: 2, sizeBytes: 2048 },
      models: { count: 0, sizeBytes: 0 },
      logs: { sizeBytes: 0 },
    })),
    executeCleanup: vi.fn(async (categories: string[]) => ({ executed: categories, modelsSkipped: [] })),
    listChats: vi.fn(async () => []),
  },
  settings: { setWebSearchModes: vi.fn(async () => undefined) },
}));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => platform }));

const cleanupDefaults = useCleanupStore.getState();

beforeEach(() => {
  vi.clearAllMocks();
  useCleanupStore.setState(cleanupDefaults, true);
  useWebSearchStore.setState({ chats: { first: "ask", second: "auto" }, last: "auto" });
});
afterEach(cleanup);

const clearDialogs = async () => {
  render(<ClearDataDialog open onClose={() => undefined} />);
  await screen.findByText(/^2 chats/);
  fireEvent.click(screen.getByRole("button", { name: "Review" }));
  fireEvent.click(screen.getByRole("button", { name: "Delete" }));
  await screen.findByText("Done");
};

it("forgets each chat's web search mode with the chats and keeps the latest one for the next chat", async () => {
  await clearDialogs();

  expect(platform.storage.executeCleanup).toHaveBeenCalledExactlyOnceWith(["dialogs"], []);
  expect(useWebSearchStore.getState()).toMatchObject({ chats: {}, last: "auto" });
  expect(platform.settings.setWebSearchModes).toHaveBeenCalledExactlyOnceWith({ chats: {}, last: "auto" });
});
