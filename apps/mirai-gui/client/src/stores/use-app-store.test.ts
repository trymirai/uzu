import { afterEach, beforeEach, expect, it, vi } from "vitest";
import bootHtml from "../../index.html?raw";
import { APP_STORE_KEY, migrateAppStorage } from "./migrate-app-storage";
import { useAppStore } from "./use-app-store";

const defaults = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(defaults, true);
  localStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

it("defaults to Medium and persists the pixel width with existing preferences", () => {
  expect(useAppStore.getState().chatWidth).toBe(1000);
  useAppStore.setState({ theme: "light", isDarkMode: false, skipWelcome: true });
  useAppStore.getState().setChatWidth(1200);
  expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state).toEqual({
    theme: "light",
    skipWelcome: true,
    chatWidth: 1200,
  });
});

it.each([800, 1000, 1200])("restores the saved %s pixel width on hydration", async (chatWidth) => {
  localStorage.setItem(APP_STORE_KEY, JSON.stringify({ state: { chatWidth }, version: 0 }));
  await useAppStore.persist.rehydrate();
  expect(useAppStore.getState().chatWidth).toBe(chatWidth);
  expect(useAppStore.getState().theme).toBe("system");
});

it.each([undefined, null, 799, 1201, "1000"])(
  "uses Medium for missing or invalid stored width %s",
  async (chatWidth) => {
    localStorage.setItem(
      APP_STORE_KEY,
      JSON.stringify({ state: { chatWidth, isDarkMode: false, skipWelcome: true }, version: 0 }),
    );
    await useAppStore.persist.rehydrate();
    expect(useAppStore.getState()).toMatchObject({ chatWidth: 1000, isDarkMode: false, skipWelcome: true });
  },
);

it.each([799, 1201, Number.NaN, Number.POSITIVE_INFINITY])("rejects invalid setter value %s", (width) => {
  useAppStore.getState().setChatWidth(width);
  expect(useAppStore.getState().chatWidth).toBe(1000);
  expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state.chatWidth).toBe(1000);
});

it("preserves legacy appearance and welcome migration while adding the wider default", async () => {
  localStorage.setItem("mirai-electron-store", JSON.stringify({ state: { isDarkMode: false }, version: 0 }));
  localStorage.setItem("mirai-auth-store", JSON.stringify({ state: { skipWelcome: true }, version: 0 }));
  migrateAppStorage();
  await useAppStore.persist.rehydrate();
  expect(useAppStore.getState()).toMatchObject({ chatWidth: 1000, isDarkMode: false, skipWelcome: true });
  expect(localStorage.getItem("mirai-electron-store")).toBeNull();
  expect(localStorage.getItem("mirai-auth-store")).toBeNull();
});

it.each([
  [{}, "system"],
  [{ isDarkMode: true }, "system"],
  [{ isDarkMode: false }, "light"],
  [{ theme: "system", isDarkMode: false }, "system"],
  [{ theme: "light" }, "light"],
  [{ theme: "dark", isDarkMode: false }, "dark"],
] as const)("keeps early boot and hydrated theme consistent for %j", async (state, expectedTheme) => {
  const script = bootHtml.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  expect(script).toBeDefined();
  for (const systemDark of [false, true]) {
    vi.stubGlobal(
      "matchMedia",
      vi.fn(() => ({ matches: systemDark })),
    );
    localStorage.setItem(APP_STORE_KEY, JSON.stringify({ state, version: 0 }));
    window.eval(script!);
    const bootIsDark = document.documentElement.classList.contains("dark");
    await useAppStore.persist.rehydrate();
    expect(useAppStore.getState().theme).toBe(expectedTheme);
    expect(useAppStore.getState().isDarkMode).toBe(
      expectedTheme === "dark" || (expectedTheme === "system" && systemDark),
    );
    expect(useAppStore.getState().isDarkMode).toBe(bootIsDark);
  }
});

it.each(["missing", "empty"])(
  "boots a legacy light preference when the current storage key is %s",
  async (currentKey) => {
    if (currentKey === "empty") localStorage.setItem(APP_STORE_KEY, "");
    vi.stubGlobal(
      "matchMedia",
      vi.fn(() => ({ matches: true })),
    );
    localStorage.setItem("mirai-electron-store", JSON.stringify({ state: { isDarkMode: false }, version: 0 }));
    window.eval(bootHtml.match(/<script>([\s\S]*?)<\/script>/)![1]!);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    migrateAppStorage();
    await useAppStore.persist.rehydrate();
    expect(useAppStore.getState()).toMatchObject({ theme: "light", isDarkMode: false });
  },
);

it("persists the selected mode without persisting the resolved system appearance", () => {
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => ({ matches: true })),
  );
  useAppStore.getState().setTheme("system");
  expect(useAppStore.getState().isDarkMode).toBe(true);
  expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state).toMatchObject({ theme: "system" });
  expect(JSON.parse(localStorage.getItem(APP_STORE_KEY)!).state).not.toHaveProperty("isDarkMode");
});
