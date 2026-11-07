import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { disableBrowserBehaviour, tauriSystem } from "./tauri";

const mocks = vi.hoisted(() => ({ isMac: false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: async () => () => {} }));
vi.mock("@/utils/platform", () => ({ isMacPlatform: () => mocks.isMac }));

const onRequest = vi.fn();
let unsubscribe = () => {};
const press = (init: KeyboardEventInit): boolean =>
  window.dispatchEvent(new KeyboardEvent("keydown", { ctrlKey: true, cancelable: true, ...init }));

beforeEach(() => {
  mocks.isMac = false;
  onRequest.mockClear();
});
afterEach(() => unsubscribe());

it("maps Ctrl+N and Ctrl+, to navigation requests, by physical key on non-Latin layouts", () => {
  unsubscribe = tauriSystem.onNavigationRequest(onRequest);
  expect(press({ key: "n", code: "KeyN" })).toBe(false);
  press({ key: ",", code: "Comma" });
  press({ key: "т", code: "KeyN" });
  press({ key: "б", code: "Comma" });
  expect(press({ key: "x", code: "KeyX" })).toBe(true);
  expect(onRequest.mock.calls.map(([request]) => request.type)).toEqual([
    "new-chat",
    "open-preferences",
    "new-chat",
    "open-preferences",
  ]);
});

it.each([{ shiftKey: true }, { altKey: true }, { metaKey: true }, { ctrlKey: false }])(
  "ignores Ctrl+N with %o",
  (init) => {
    unsubscribe = tauriSystem.onNavigationRequest(onRequest);
    press({ key: "n", code: "KeyN", ...init });
    expect(onRequest).not.toHaveBeenCalled();
  },
);

it("leaves the shortcuts to the macOS app menu", () => {
  mocks.isMac = true;
  unsubscribe = tauriSystem.onNavigationRequest(onRequest);
  press({ key: "n", code: "KeyN" });
  expect(onRequest).not.toHaveBeenCalled();
});

describe("disableBrowserBehaviour", () => {
  beforeAll(() => {
    mocks.isMac = false;
    vi.stubEnv("PROD", true);
    disableBrowserBehaviour();
    vi.unstubAllEnvs();
  });

  it("blocks Ctrl+R, by physical key on non-Latin layouts", () => {
    expect(press({ key: "r", code: "KeyR" })).toBe(false);
    expect(press({ key: "к", code: "KeyR" })).toBe(false);
  });

  it("lets AltGr, which WebView2 reports as Ctrl+Alt, type text", () => {
    expect(press({ key: "ś", code: "KeyS", altKey: true })).toBe(true);
    expect(press({ key: "[", code: "KeyF", altKey: true })).toBe(true);
    expect(press({ key: "]", code: "KeyG", altKey: true })).toBe(true);
  });

  it("keeps the native context menu in text fields only", () => {
    document.body.innerHTML = `<input /><textarea></textarea><div contenteditable="true"><p></p></div><button></button>`;
    const rightClick = (selector: string): boolean =>
      document
        .querySelector(selector)
        ?.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true })) ?? false;
    expect(["input", "textarea", "[contenteditable] p", "button"].map(rightClick)).toEqual([true, true, true, false]);
  });
});
