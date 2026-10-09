import { beforeEach, expect, it, vi } from "vitest";
import { useSettingsStore } from "./use-settings-store";

const settings = vi.hoisted(() => ({
  getAnalyticsEnabled: vi.fn(async () => false),
  setAnalyticsEnabled: vi.fn<(enabled: boolean) => Promise<void>>().mockResolvedValue(undefined),
  getModelChatNamingEnabled: vi.fn(async () => true),
  setModelChatNamingEnabled: vi.fn<(enabled: boolean) => Promise<void>>().mockResolvedValue(undefined),
  getAutoEjectEnabled: vi.fn(async () => true),
  getAutoEjectMinutes: vi.fn(async () => 15),
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
