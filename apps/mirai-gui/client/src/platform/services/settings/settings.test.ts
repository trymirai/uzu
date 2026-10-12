import { beforeEach, describe, expect, it, vi } from "vitest";
import { tauriSettings } from "./tauri";
import { webSettings } from "./web";
import { invoke } from "../shared/invoke";

vi.mock("../shared/invoke", () => ({ invoke: vi.fn() }));

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
});

describe("analytics consent", () => {
  it.each([undefined, null, false, "true", 1])("requires an explicit desktop opt-in, not %s", async (value) => {
    vi.mocked(invoke).mockResolvedValue({ analyticsEnabled: value });
    expect(await tauriSettings.getAnalyticsEnabled()).toBe(false);
  });

  it("loads and patches explicit desktop consent", async () => {
    vi.mocked(invoke).mockResolvedValue({ analyticsEnabled: true });
    expect(await tauriSettings.getAnalyticsEnabled()).toBe(true);

    await tauriSettings.setAnalyticsEnabled(false);
    expect(invoke).toHaveBeenLastCalledWith("settings_patch", { patch: { analyticsEnabled: false } });
  });

  it("defaults off in the browser and preserves other preferences when changed", async () => {
    localStorage.setItem("mirai.web.settings", JSON.stringify({ modelChatNamingEnabled: false }));
    expect(await webSettings.getAnalyticsEnabled()).toBe(false);

    await webSettings.setAnalyticsEnabled(true);
    expect(await webSettings.getAnalyticsEnabled()).toBe(true);
    expect(await webSettings.getModelChatNamingEnabled()).toBe(false);
    await webSettings.setAnalyticsEnabled(false);
    expect(await webSettings.getAnalyticsEnabled()).toBe(false);
  });
});

describe("model chat naming preference", () => {
  it("defaults on for existing desktop settings", async () => {
    vi.mocked(invoke).mockResolvedValue({ enableThinking: false });
    expect(await tauriSettings.getModelChatNamingEnabled()).toBe(true);
  });

  it("loads and patches the disabled desktop preference", async () => {
    vi.mocked(invoke).mockResolvedValue({ modelChatNamingEnabled: false });
    expect(await tauriSettings.getModelChatNamingEnabled()).toBe(false);

    await tauriSettings.setModelChatNamingEnabled(false);
    expect(invoke).toHaveBeenLastCalledWith("settings_patch", { patch: { modelChatNamingEnabled: false } });
  });

  it("defaults on in the browser and persists disabling without replacing other settings", async () => {
    localStorage.setItem("mirai.web.settings", JSON.stringify({ unrelatedPreference: true }));
    expect(await webSettings.getModelChatNamingEnabled()).toBe(true);

    await webSettings.setModelChatNamingEnabled(false);
    expect(await webSettings.getModelChatNamingEnabled()).toBe(false);
    expect(JSON.parse(localStorage.getItem("mirai.web.settings")!)).toMatchObject({ unrelatedPreference: true });
  });
});

describe("auto-eject defaults", () => {
  it("defaults to fifteen minutes on desktop and web", async () => {
    vi.mocked(invoke).mockResolvedValue({});
    expect(await tauriSettings.getAutoEjectMinutes()).toBe(15);
    expect(await webSettings.getAutoEjectMinutes()).toBe(15);
    expect(await tauriSettings.getAutoEjectEnabled()).toBe(true);
  });

  it.each([2, 7, 30])("preserves a saved interval of %s minutes", async (minutes) => {
    vi.mocked(invoke).mockResolvedValue({ autoEjectMinutes: minutes, autoEjectEnabled: false });
    expect(await tauriSettings.getAutoEjectMinutes()).toBe(minutes);
    expect(await tauriSettings.getAutoEjectEnabled()).toBe(false);
  });

  it.each([null, "7", 0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
    "uses the default for invalid interval %s",
    async (minutes) => {
      vi.mocked(invoke).mockResolvedValue({ autoEjectMinutes: minutes });
      expect(await tauriSettings.getAutoEjectMinutes()).toBe(15);
    },
  );
});
