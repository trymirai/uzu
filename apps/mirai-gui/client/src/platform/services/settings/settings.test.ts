import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tauriSettings } from "./tauri";
import { onBrowserAutoEjectChanged, webSettings } from "./web";
import { invoke } from "../shared/invoke";

vi.mock("../shared/invoke", () => ({ invoke: vi.fn() }));

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
});
afterEach(() => vi.restoreAllMocks());

describe.each(["QuotaExceededError", "SecurityError"])("browser persistence failure: %s", (name) => {
  const changes: Array<[string, () => Promise<unknown>]> = [
    ["analytics opt-out", () => webSettings.setAnalyticsEnabled(false)],
    ["chat naming", () => webSettings.setModelChatNamingEnabled(false)],
    ["model parameters", () => webSettings.setModelParams("model", { sampling: { type: "Default" } })],
    ["reset model parameters", () => webSettings.setModelParams("model", null)],
    ["auto-eject enabled", () => webSettings.setAutoEjectEnabled(false)],
    ["auto-eject interval", () => webSettings.setAutoEjectMinutes(2)],
  ];

  it.each(changes)("rejects %s without changing saved values or idle timers", async (_, change) => {
    const preferences = JSON.stringify({ autoEjectEnabled: true, autoEjectMinutes: 15, modelChatNamingEnabled: true });
    const parameters = JSON.stringify({ model: { sampling: { type: "Greedy" } } });
    localStorage.setItem("mirai.web.settings", preferences);
    localStorage.setItem("mirai.web.modelParams", parameters);
    const notify = vi.fn();
    const stop = onBrowserAutoEjectChanged(notify);
    const error = new DOMException("Could not save browser settings", name);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw error;
    });
    try {
      const result = change();
      expect(result).toBeInstanceOf(Promise);
      await expect(result).rejects.toBe(error);
      expect(localStorage.getItem("mirai.web.settings")).toBe(preferences);
      expect(localStorage.getItem("mirai.web.modelParams")).toBe(parameters);
      expect(notify).not.toHaveBeenCalled();
    } finally {
      stop();
    }
  });
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

  it("does not claim analytics is enabled in the browser or overwrite other preferences", async () => {
    localStorage.setItem(
      "mirai.web.settings",
      JSON.stringify({ analyticsEnabled: true, modelChatNamingEnabled: false }),
    );
    expect(await webSettings.getAnalyticsEnabled()).toBe(false);

    await expect(webSettings.setAnalyticsEnabled(true)).rejects.toThrow("unavailable");
    expect(await webSettings.getAnalyticsEnabled()).toBe(false);
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

it("restores browser tool and reasoning preferences without overriding explicit choices", async () => {
  localStorage.setItem(
    "mirai.web.modelParams",
    JSON.stringify({
      model: {
        sampling: { type: "Greedy" },
        reasoningEffort: "disabled",
        dateTimeToolEnabled: true,
        chartToolEnabled: true,
        modelChatNamingEnabled: true,
      },
    }),
  );
  expect((await webSettings.getModelParams()).model).toEqual({
    sampling: { type: "Greedy" },
    reasoningEffort: "disabled",
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
    modelChatNamingEnabled: true,
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
