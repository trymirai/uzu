import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import GeneralTab from "./general-tab";

vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({ system: { getCliStatus: () => Promise.resolve("installed") } }),
}));

vi.mock("@/platform/platform-info", () => ({
  platformInfo: { isTauri: true, features: { globalShortcut: true } },
}));

const settings = {
  enableThinking: true,
  runOnStartup: false,
  quickEntryShortcut: "CommandOrControl+Shift+M",
  autoEjectEnabled: true,
  autoEjectMinutes: 7,
  fetch: vi.fn(),
  fetchDesktopSettings: vi.fn(() => Promise.resolve()),
  setEnableThinking: vi.fn(),
  setRunOnStartup: vi.fn(),
  registerQuickEntryShortcut: vi.fn(() => Promise.resolve(true)),
  unregisterQuickEntryShortcut: vi.fn(() => Promise.resolve()),
  setAutoEjectEnabled: vi.fn(),
  setAutoEjectMinutes: vi.fn(),
};

vi.mock("@/stores/use-settings-store", () => ({ useSettingsStore: () => settings }));

vi.mock("@/stores/use-global-instructions-store", () => ({
  useGlobalInstructionsStore: (selector: (s: unknown) => unknown) =>
    selector({ instructions: "", loadInstructions: vi.fn(), saveInstructions: vi.fn() }),
}));

vi.mock("@/stores/use-app-store", () => ({
  useAppStore: (selector: (s: unknown) => unknown) => selector({ isDarkMode: true, setDarkMode: vi.fn() }),
}));

describe("quick entry shortcut capture", () => {
  it("registers a combination with a system modifier and ignores one without", async () => {
    render(<GeneralTab />);
    fireEvent.click(screen.getByRole("button", { name: /⌘ ⇧ M/ }));
    await new Promise((r) => setTimeout(r, 0));

    fireEvent.keyDown(window, { key: "k", code: "KeyK", metaKey: true, shiftKey: true });
    expect(settings.registerQuickEntryShortcut).toHaveBeenCalledWith("CommandOrControl+Shift+K");

    settings.registerQuickEntryShortcut.mockClear();
    fireEvent.keyDown(window, { key: "j", code: "KeyJ", shiftKey: true });
    expect(settings.registerQuickEntryShortcut).not.toHaveBeenCalled();
  });
});
