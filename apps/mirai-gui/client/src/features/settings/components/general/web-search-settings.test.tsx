import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { WebSearchKeyProvider, WebSearchProvider } from "@/constants/web-search-providers";
import { WebSearchSettings } from "./web-search-settings";

const settings = vi.hoisted(() => ({
  webSearchProvider: "duckduckgo" as WebSearchProvider,
  webSearchKeys: {} as Partial<Record<WebSearchKeyProvider, string>>,
  setWebSearchProvider: vi.fn<(provider: WebSearchProvider) => Promise<void>>(async () => undefined),
  saveWebSearchKey: vi.fn<(provider: WebSearchKeyProvider, key: string) => Promise<string | null>>(async () => null),
  removeWebSearchKey: vi.fn<(provider: WebSearchKeyProvider) => Promise<boolean>>(async () => true),
}));
vi.mock("@/stores/use-settings-store", () => ({ useSettingsStore: () => settings }));
const toast = vi.hoisted(() => ({ error: vi.fn() }));
vi.mock("@/components/ui/toast/use-toast", () => ({ useToast: () => toast }));

beforeEach(() => {
  vi.clearAllMocks();
  settings.webSearchProvider = "duckduckgo";
  settings.webSearchKeys = {};
});
afterEach(cleanup);

const enterKey = async (key: string) => {
  const field = await screen.findByPlaceholderText("Paste your key");
  fireEvent.change(field, { target: { value: key } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
};

it("removes a key only after the user confirms, and says when the provider will change", async () => {
  settings.webSearchProvider = "brave";
  settings.webSearchKeys = { brave: "a1b2" };
  settings.removeWebSearchKey.mockResolvedValueOnce(false);
  render(<WebSearchSettings />);

  fireEvent.click(screen.getByRole("button", { name: "Remove Brave Search API key" }));
  expect(settings.removeWebSearchKey).not.toHaveBeenCalled();
  expect(await screen.findByText(/Web search will switch to DuckDuckGo\./)).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Remove" }));
  expect(settings.removeWebSearchKey).toHaveBeenCalledExactlyOnceWith("brave");
  await waitFor(() => expect(toast.error).toHaveBeenCalledExactlyOnceWith("Failed to remove the key"));
});

it("asks for the key before Brave becomes the provider, and chooses Brave once the key is saved", async () => {
  render(<WebSearchSettings />);
  fireEvent.click(screen.getByRole("button", { name: "Search provider: DuckDuckGo" }));
  fireEvent.click(await screen.findByRole("option", { name: "Brave" }));
  expect(settings.setWebSearchProvider).not.toHaveBeenCalled();

  await enterKey("brave-key");
  await waitFor(() => expect(settings.setWebSearchProvider).toHaveBeenCalledExactlyOnceWith("brave"));
  expect(settings.saveWebSearchKey).toHaveBeenCalledExactlyOnceWith("brave", "brave-key");
});
