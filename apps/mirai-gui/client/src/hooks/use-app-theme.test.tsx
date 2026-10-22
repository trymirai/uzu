import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAppStore } from "@/stores/use-app-store";
import { useAppTheme } from "./use-app-theme";
import { useAppInitialization } from "./use-app-initialization";

const setWindowTheme = vi.hoisted(() => vi.fn(async () => true));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ systemUi: { setWindowTheme } }) }));
vi.mock("./session-wiring", () => ({ useSessionWiring: vi.fn() }));
vi.mock("./use-download-wiring", () => ({ useDownloadWiring: vi.fn() }));
vi.mock("./use-update-initialization", () => ({ useUpdateInitialization: vi.fn() }));
const defaults = useAppStore.getState();
let systemDark = false;
let listeners: Set<(event: { matches: boolean }) => void>;

beforeEach(() => {
  vi.clearAllMocks();
  systemDark = false;
  listeners = new Set();
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => ({
      get matches() {
        return systemDark;
      },
      addEventListener: (_name: string, listener: (event: { matches: boolean }) => void) => listeners.add(listener),
      removeEventListener: (_name: string, listener: (event: { matches: boolean }) => void) =>
        listeners.delete(listener),
    })),
  );
  useAppStore.setState({ ...defaults, theme: "system", isDarkMode: false }, true);
  document.documentElement.classList.remove("dark");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  document.documentElement.classList.remove("dark");
});

const changeSystemTheme = (dark: boolean) =>
  act(() => {
    systemDark = dark;
    listeners.forEach((listener) => listener({ matches: dark }));
  });

it("follows live OS changes by default and leaves the native window in System mode", () => {
  const hook = renderHook(useAppTheme);
  expect(useAppStore.getState().theme).toBe("system");
  expect(document.documentElement.classList.contains("dark")).toBe(false);
  expect(setWindowTheme).toHaveBeenCalledWith("system");

  changeSystemTheme(true);
  expect(document.documentElement.classList.contains("dark")).toBe(true);
  expect(useAppStore.getState().isDarkMode).toBe(true);
  changeSystemTheme(false);
  expect(document.documentElement.classList.contains("dark")).toBe(false);
  expect(useAppStore.getState().isDarkMode).toBe(false);
  expect(setWindowTheme).toHaveBeenCalledOnce();
  hook.unmount();
  expect(listeners.size).toBe(0);
});

it("keeps explicit Light and Dark modes through OS changes, then resumes System mode", () => {
  renderHook(useAppTheme);
  act(() => useAppStore.getState().setTheme("dark"));
  expect(document.documentElement.classList.contains("dark")).toBe(true);
  expect(setWindowTheme).toHaveBeenLastCalledWith("dark");
  changeSystemTheme(true);
  changeSystemTheme(false);
  expect(useAppStore.getState().isDarkMode).toBe(true);

  act(() => useAppStore.getState().setTheme("light"));
  changeSystemTheme(true);
  expect(document.documentElement.classList.contains("dark")).toBe(false);
  expect(useAppStore.getState().isDarkMode).toBe(false);
  expect(setWindowTheme).toHaveBeenLastCalledWith("light");

  act(() => useAppStore.getState().setTheme("system"));
  expect(document.documentElement.classList.contains("dark")).toBe(true);
  expect(useAppStore.getState().isDarkMode).toBe(true);
  expect(setWindowTheme).toHaveBeenLastCalledWith("system");
});

it("still follows the system on the welcome screen while app initialization is disabled", () => {
  renderHook(() => useAppInitialization(false));
  changeSystemTheme(true);
  expect(document.documentElement.classList.contains("dark")).toBe(true);
  expect(useAppStore.getState().isDarkMode).toBe(true);
  expect(setWindowTheme).toHaveBeenCalledWith("system");
});
