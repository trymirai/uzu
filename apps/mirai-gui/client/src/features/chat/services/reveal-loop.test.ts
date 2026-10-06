import { afterEach, expect, it, vi } from "vitest";
import { createRevealLoop } from "./reveal-loop";

const setVisibility = (state: DocumentVisibilityState) =>
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => state });

afterEach(() => {
  setVisibility("visible");
  vi.restoreAllMocks();
});

it("shows everything at once and settles drain while the window is hidden", async () => {
  setVisibility("hidden");
  const raf = vi.spyOn(window, "requestAnimationFrame");
  const applied: string[] = [];
  const loop = createRevealLoop({ baseText: "", isActive: () => true, apply: (text) => applied.push(text) });

  loop.append("Hello");
  loop.append(" world");
  await loop.drain();

  expect(applied.at(-1)).toBe("Hello world");
  expect(raf).not.toHaveBeenCalled();
});

it("settles drain when the window is hidden after a frame was already requested", async () => {
  const raf = vi.spyOn(window, "requestAnimationFrame").mockReturnValue(1);
  const caf = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
  const applied: string[] = [];
  const loop = createRevealLoop({ baseText: "", isActive: () => true, apply: (text) => applied.push(text) });

  loop.append("Hello");
  expect(raf).toHaveBeenCalledTimes(1);
  setVisibility("hidden");
  loop.append(" world");
  await loop.drain();

  expect(caf).toHaveBeenCalledWith(1);
  expect(applied.at(-1)).toBe("Hello world");
});

it("settles a drain already waiting for a frame when the window is hidden", async () => {
  vi.spyOn(window, "requestAnimationFrame").mockReturnValue(1);
  const caf = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
  const applied: string[] = [];
  const loop = createRevealLoop({ baseText: "", isActive: () => true, apply: (text) => applied.push(text) });

  loop.append("Hello world");
  const drained = loop.drain();
  setVisibility("hidden");
  document.dispatchEvent(new Event("visibilitychange"));
  await drained;

  expect(caf).toHaveBeenCalledWith(1);
  expect(applied.at(-1)).toBe("Hello world");
});
