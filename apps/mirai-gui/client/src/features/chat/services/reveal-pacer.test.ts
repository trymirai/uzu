import { afterEach, describe, expect, it, vi } from "vitest";
import { RevealPacer } from "./reveal-pacer";
import { createRevealLoop } from "./reveal-loop";

describe("RevealPacer", () => {
  it("holds text back during warm-up, then paces it at the measured rate", () => {
    const pacer = new RevealPacer("");
    pacer.append("hello");

    expect(pacer.step(100)).toBeNull();
    expect(pacer.step(100)).toBe("he");
    expect(pacer.isCaughtUp).toBe(false);
  });

  it("skips the warm-up once the stream is done", () => {
    const pacer = new RevealPacer("");
    pacer.append("hi");
    pacer.markDone();

    expect(pacer.step(50)).toBe("hi");
    expect(pacer.isCaughtUp).toBe(true);
  });

  it("caps the reveal rate at 600 characters per second", () => {
    const pacer = new RevealPacer("");
    pacer.append("x".repeat(10_000));

    expect(pacer.step(200)?.length).toBe(120);
  });

  it("keeps the base text visible and paces only what arrived after it", () => {
    const pacer = new RevealPacer("abc");
    pacer.append("def");

    expect(pacer.step(200)).toBe("abcdef");
  });
});

describe("createRevealLoop", () => {
  const frames: Array<(now: number) => void> = [];
  let now = 0;
  const runFrame = () => {
    now += 16;
    const pending = frames.splice(0, frames.length);
    pending.forEach((cb) => cb(now));
  };

  afterEach(() => {
    vi.unstubAllGlobals();
    frames.length = 0;
    now = 0;
  });

  it("shows the whole text before drain resolves", async () => {
    vi.stubGlobal("requestAnimationFrame", (cb: (now: number) => void) => frames.push(cb));
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    const shown: string[] = [];
    const loop = createRevealLoop({ baseText: "", isActive: () => true, apply: (text) => shown.push(text) });

    loop.append("streamed reply");
    const drained = loop.drain();
    for (let i = 0; i < 100 && frames.length > 0; i += 1) runFrame();
    await drained;

    expect(shown.at(-1)).toBe("streamed reply");
    expect(frames).toHaveLength(0);
  });

  it("stops applying once another run took over", async () => {
    vi.stubGlobal("requestAnimationFrame", (cb: (now: number) => void) => frames.push(cb));
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    const shown: string[] = [];
    let active = true;
    const loop = createRevealLoop({ baseText: "", isActive: () => active, apply: (text) => shown.push(text) });

    loop.append("first");
    active = false;
    const drained = loop.drain();
    runFrame();
    await drained;

    expect(shown).toEqual([]);
  });
});
