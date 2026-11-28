import { beforeEach, expect, it, vi } from "vitest";
import type { WebSearchKeyProvider, WebSearchProvider } from "@/constants/web-search-providers";
import type { WebSearchStoredKey } from "@/types/web-search";
import { useSettingsStore } from "./use-settings-store";

const settings = vi.hoisted(() => ({
  getAnalyticsEnabled: vi.fn(async () => false),
  setAnalyticsEnabled: vi.fn<(enabled: boolean) => Promise<void>>().mockResolvedValue(undefined),
  getModelChatNamingEnabled: vi.fn(async () => true),
  setModelChatNamingEnabled: vi.fn<(enabled: boolean) => Promise<void>>().mockResolvedValue(undefined),
  getAutoEjectEnabled: vi.fn(async () => true),
  getAutoEjectMinutes: vi.fn(async () => 15),
  getWebSearchProvider: vi.fn(async (): Promise<WebSearchProvider> => "exa"),
  setWebSearchProvider: vi.fn<(provider: WebSearchProvider) => Promise<void>>().mockResolvedValue(undefined),
  getWebSearchKeys: vi.fn(async (): Promise<WebSearchStoredKey[]> => []),
  saveWebSearchKey: vi.fn<(provider: WebSearchKeyProvider, key: string) => Promise<WebSearchStoredKey[]>>(),
  removeWebSearchKey: vi.fn<(provider: WebSearchKeyProvider) => Promise<WebSearchStoredKey[]>>(),
}));

vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings }) }));

const defaults = useSettingsStore.getState();

beforeEach(() => {
  vi.clearAllMocks();
  useSettingsStore.setState(defaults, true);
});

it("starts with analytics disabled and stays disabled when settings cannot be read", async () => {
  expect(useSettingsStore.getState().analyticsEnabled).toBe(false);
  settings.getAnalyticsEnabled.mockRejectedValueOnce(new Error("read failed"));

  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().analyticsEnabled).toBe(false);
});

it("loads and saves explicit analytics consent", async () => {
  settings.getAnalyticsEnabled.mockResolvedValueOnce(true);
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().analyticsEnabled).toBe(true);

  await useSettingsStore.getState().setAnalyticsEnabled(false);
  expect(settings.setAnalyticsEnabled).toHaveBeenCalledWith(false);
  expect(useSettingsStore.getState().analyticsEnabled).toBe(false);
});

it("does not show analytics as enabled when saving consent fails", async () => {
  settings.setAnalyticsEnabled.mockRejectedValueOnce(new Error("Could not save settings"));

  await useSettingsStore.getState().setAnalyticsEnabled(true);
  expect(useSettingsStore.getState().analyticsEnabled).toBe(false);
});

it("starts with chat naming enabled and loads a saved opt-out", async () => {
  expect(useSettingsStore.getState().modelChatNamingEnabled).toBe(true);
  settings.getModelChatNamingEnabled.mockResolvedValueOnce(false);

  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().modelChatNamingEnabled).toBe(false);
});

it("keeps chat naming disabled after successfully saving the preference", async () => {
  await useSettingsStore.getState().setModelChatNamingEnabled(false);

  expect(settings.setModelChatNamingEnabled).toHaveBeenCalledWith(false);
  expect(useSettingsStore.getState().modelChatNamingEnabled).toBe(false);
});

it("restores the previous preference if saving fails", async () => {
  settings.setModelChatNamingEnabled.mockRejectedValueOnce(new Error("Could not save settings"));

  await useSettingsStore.getState().setModelChatNamingEnabled(false);
  expect(useSettingsStore.getState().modelChatNamingEnabled).toBe(true);
});

it("defaults auto-eject to fifteen minutes, including when settings cannot be read", async () => {
  expect(useSettingsStore.getState().autoEjectMinutes).toBe(15);
  settings.getAutoEjectMinutes.mockRejectedValueOnce(new Error("read failed"));
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().autoEjectMinutes).toBe(15);
});

it("loads an explicit auto-eject interval without replacing it with the default", async () => {
  settings.getAutoEjectMinutes.mockResolvedValueOnce(7);
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().autoEjectMinutes).toBe(7);
});

it("reads the saved search provider and falls back to DuckDuckGo when it cannot be read", async () => {
  settings.getWebSearchProvider.mockResolvedValueOnce("keenable");
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().webSearchProvider).toBe("keenable");

  settings.getWebSearchProvider.mockRejectedValueOnce(new Error("read failed"));
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().webSearchProvider).toBe("duckduckgo");
});

it("applies a search provider choice at once and restores the previous one when it cannot be saved", async () => {
  await useSettingsStore.getState().setWebSearchProvider("keenable");
  expect(settings.setWebSearchProvider).toHaveBeenCalledWith("keenable");
  expect(useSettingsStore.getState().webSearchProvider).toBe("keenable");

  settings.setWebSearchProvider.mockRejectedValueOnce(new Error("write failed"));
  await useSettingsStore.getState().setWebSearchProvider("exa");
  expect(useSettingsStore.getState().webSearchProvider).toBe("keenable");
});

it("shows how the stored keys end, saves a key the provider accepts and reports one it refuses", async () => {
  settings.getWebSearchKeys.mockResolvedValueOnce([{ provider: "exa", ending: "9f2c" }]);
  await useSettingsStore.getState().fetch();
  expect(useSettingsStore.getState().webSearchKeys).toEqual({ exa: "9f2c" });

  settings.saveWebSearchKey.mockRejectedValueOnce(new Error("Brave did not accept this key"));
  expect(await useSettingsStore.getState().saveWebSearchKey("brave", "wrong")).toBe("Brave did not accept this key");
  expect(useSettingsStore.getState().webSearchKeys).toEqual({ exa: "9f2c" });

  settings.saveWebSearchKey.mockResolvedValueOnce([
    { provider: "brave", ending: "a1b2" },
    { provider: "exa", ending: "9f2c" },
  ]);
  expect(await useSettingsStore.getState().saveWebSearchKey("brave", "right")).toBeNull();
  expect(settings.saveWebSearchKey).toHaveBeenLastCalledWith("brave", "right");
  expect(useSettingsStore.getState().webSearchKeys).toEqual({ brave: "a1b2", exa: "9f2c" });
});

it("goes back to DuckDuckGo when the key of the chosen provider that needs one is removed", async () => {
  useSettingsStore.setState({ webSearchProvider: "exa", webSearchKeys: { brave: "a1b2", exa: "9f2c" } });
  settings.removeWebSearchKey.mockResolvedValueOnce([{ provider: "brave", ending: "a1b2" }]);
  await useSettingsStore.getState().removeWebSearchKey("exa");
  expect(useSettingsStore.getState()).toMatchObject({ webSearchProvider: "exa", webSearchKeys: { brave: "a1b2" } });

  useSettingsStore.setState({ webSearchProvider: "brave" });
  settings.removeWebSearchKey.mockResolvedValueOnce([]);
  await useSettingsStore.getState().removeWebSearchKey("brave");
  expect(useSettingsStore.getState()).toMatchObject({ webSearchProvider: "duckduckgo", webSearchKeys: {} });
  expect(settings.setWebSearchProvider).toHaveBeenCalledExactlyOnceWith("duckduckgo");

  useSettingsStore.setState({ webSearchProvider: "brave", webSearchKeys: { brave: "a1b2" } });
  settings.removeWebSearchKey.mockRejectedValueOnce(new Error("keychain"));
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  await useSettingsStore.getState().removeWebSearchKey("brave");
  expect(useSettingsStore.getState()).toMatchObject({ webSearchProvider: "brave", webSearchKeys: { brave: "a1b2" } });
});
