import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAppStore } from "@/stores/use-app-store";
import { APP_STORE_KEY } from "@/stores/migrate-app-storage";
import GeneralTab from "./general-tab";

vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({ system: { getCliStatus: () => Promise.resolve("installed") } }),
}));

vi.mock("@/platform/platform-info", () => ({
  platformInfo: { isTauri: true, features: { logExport: true } },
}));

const settings = {
  analyticsEnabled: false,
  modelChatNamingEnabled: true,
  autoEjectEnabled: true,
  autoEjectMinutes: 7,
  fetch: vi.fn(),
  setAnalyticsEnabled: vi.fn(),
  setModelChatNamingEnabled: vi.fn(),
  setAutoEjectEnabled: vi.fn(),
  setAutoEjectMinutes: vi.fn(),
  exportLogs: vi.fn(async () => "ok"),
};

vi.mock("@/stores/use-settings-store", () => ({ useSettingsStore: () => settings }));

const exportChats = vi.hoisted(() => vi.fn(async () => true));
vi.mock("@/features/chat/services/export-chats", () => ({ exportAllChatsZip: exportChats }));
vi.mock("./clear-data-dialog", () => ({
  ClearDataDialog: ({ open }: { open: boolean }) => (open ? <div role="dialog">Clear data confirmation</div> : null),
}));

vi.mock("@/stores/use-global-instructions-store", () => ({
  useGlobalInstructionsStore: (selector: (s: unknown) => unknown) =>
    selector({ instructions: "", loadInstructions: vi.fn(), saveInstructions: vi.fn() }),
}));

const appDefaults = useAppStore.getState();

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
  useAppStore.setState(appDefaults, true);
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("lets users disable model-provided chat names", () => {
  render(<GeneralTab />);
  const toggle = screen.getByRole("switch", { name: "Let models name chats via tool call" });
  expect(toggle.getAttribute("aria-checked")).toBe("true");

  fireEvent.click(toggle);
  expect(settings.setModelChatNamingEnabled).toHaveBeenCalledWith(false);
});

it("leaves usage analytics off until the user opts in", () => {
  render(<GeneralTab />);
  const toggle = screen.getByRole("switch", { name: "Share usage analytics" });
  expect(toggle.getAttribute("aria-checked")).toBe("false");
  expect(settings.setAnalyticsEnabled).not.toHaveBeenCalled();

  fireEvent.click(toggle);
  expect(settings.setAnalyticsEnabled).toHaveBeenCalledWith(true);
});

it("opens analytics details in a popover that dismisses with Escape or an outside click", async () => {
  render(<GeneralTab />);
  const more = screen.getByRole("button", { name: "See more" });
  expect(more.getAttribute("aria-expanded")).toBe("false");
  expect(screen.queryByRole("region", { name: "Usage analytics details" })).toBeNull();

  fireEvent.click(more);
  const details = await screen.findByRole("region", { name: "Usage analytics details" });
  expect(more.getAttribute("aria-expanded")).toBe("true");
  expect(more.getAttribute("aria-controls")).toBe(details.id);
  expect(within(details).getAllByRole("listitem")).toHaveLength(5);
  expect(within(details).getByText(/Chat content, attachments, chat names, local paths/).textContent).toContain(
    "connection’s IP address",
  );
  expect(settings.setAnalyticsEnabled).not.toHaveBeenCalled();

  await waitFor(() => expect(details.contains(document.activeElement)).toBe(true));
  fireEvent.keyDown(document.activeElement!, { key: "Escape" });
  await waitFor(() => expect(screen.queryByRole("region", { name: "Usage analytics details" })).toBeNull());
  expect(more.getAttribute("aria-expanded")).toBe("false");
  expect(document.activeElement).toBe(more);
  expect(more.hasAttribute("data-focus")).toBe(true);

  fireEvent.click(more);
  await screen.findByRole("region", { name: "Usage analytics details" });
  const outside = screen.getByRole("heading", { name: "Theme" });
  fireEvent.pointerDown(outside);
  fireEvent.pointerUp(outside);
  fireEvent.click(outside);
  await waitFor(() => expect(screen.queryByRole("region", { name: "Usage analytics details" })).toBeNull());
  expect(more.getAttribute("aria-expanded")).toBe("false");
});

it("keeps focus without a keyboard ring when See more is closed by clicking it again", async () => {
  render(<GeneralTab />);
  const more = screen.getByRole("button", { name: "See more" });
  const clickWithMouse = () => {
    fireEvent.pointerDown(more, { pointerType: "mouse" });
    fireEvent.mouseDown(more);
    fireEvent.pointerUp(more, { pointerType: "mouse" });
    fireEvent.mouseUp(more);
    fireEvent.click(more, { detail: 1 });
  };

  clickWithMouse();
  const details = await screen.findByRole("region", { name: "Usage analytics details" });
  await waitFor(() => expect(details.contains(document.activeElement)).toBe(true));

  clickWithMouse();
  await waitFor(() => expect(screen.queryByRole("region", { name: "Usage analytics details" })).toBeNull());
  expect(document.activeElement).toBe(more);
  expect(more.hasAttribute("data-focus")).toBe(false);
});

it("makes exports and data cleanup available in General settings", async () => {
  render(<GeneralTab />);
  const chatsRow = screen.getByRole("heading", { name: "Export all your chats" }).closest(".justify-between")!;
  const logsRow = screen.getByRole("heading", { name: "Export logs" }).closest(".justify-between")!;
  fireEvent.click(within(chatsRow as HTMLElement).getByRole("button", { name: "Export" }));
  fireEvent.click(within(logsRow as HTMLElement).getByRole("button", { name: "Export" }));
  await waitFor(() => {
    expect(exportChats).toHaveBeenCalledOnce();
    expect(settings.exportLogs).toHaveBeenCalledOnce();
  });
  expect(screen.queryByRole("dialog")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Clear data" }));
  expect(screen.getByRole("dialog").textContent).toBe("Clear data confirmation");
});

it("updates and persists chat width from General settings", () => {
  render(<GeneralTab />);
  expect(screen.getByRole("radio", { name: "Medium" }).getAttribute("aria-checked")).toBe("true");

  for (const [label, width] of [
    ["Narrow", 800],
    ["Wide", 1200],
    ["Medium", 1000],
  ] as const) {
    const option = screen.getByRole("radio", { name: label });
    fireEvent.click(option);
    expect(option.getAttribute("aria-checked")).toBe("true");
    expect(useAppStore.getState().chatWidth).toBe(width);
    expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state.chatWidth).toBe(width);
  }
});

it("persists System, Light, and Dark choices through the custom theme selector", async () => {
  render(<GeneralTab />);
  let selectedLabel = "System";
  for (const [mode, label] of [
    ["light", "Light"],
    ["dark", "Dark"],
    ["system", "System"],
  ] as const) {
    fireEvent.click(screen.getByRole("button", { name: `Theme: ${selectedLabel}` }));
    const popup = await screen.findByRole("listbox", { name: `Theme: ${selectedLabel}` });
    expect(screen.getByRole("option", { name: selectedLabel }).getAttribute("aria-selected")).toBe("true");
    const option = screen.getByRole("option", { name: label });
    if (mode === "dark") {
      fireEvent.keyDown(popup, { key: "End" });
      await waitFor(() => expect(popup.getAttribute("aria-activedescendant")).toBe(option.id));
      fireEvent.keyDown(popup, { key: "Enter" });
    } else {
      fireEvent.click(option);
    }
    expect(screen.getByRole("button", { name: `Theme: ${label}` })).toBeTruthy();
    expect(useAppStore.getState().theme).toBe(mode);
    expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state.theme).toBe(mode);
    selectedLabel = label;
  }
});
