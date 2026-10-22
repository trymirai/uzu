import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ThinkingBlock } from "./thinking-block";

vi.mock("./markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}));

let observers: Set<{ elements: Set<Element>; callback: () => void }>;
const originalGetAnimations = Object.getOwnPropertyDescriptor(Element.prototype, "getAnimations");

beforeEach(() => {
  Object.defineProperty(Element.prototype, "getAnimations", { configurable: true, value: () => [] });
  observers = new Set();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      elements = new Set<Element>();
      constructor(public callback: () => void) {
        observers.add(this);
      }
      observe(element: Element) {
        this.elements.add(element);
      }
      disconnect() {
        observers.delete(this);
      }
    },
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  if (originalGetAnimations) Object.defineProperty(Element.prototype, "getAnimations", originalGetAnimations);
  else delete (Element.prototype as Partial<Element>).getAnimations;
});

const resize = (element: Element) =>
  act(() => {
    observers.forEach((observer) => {
      if (observer.elements.has(element)) observer.callback();
    });
  });

const settleScrolling = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }),
  );

const measure = (element: HTMLDivElement, height: number, contentHeight: number) => {
  const geometry = { height, contentHeight, top: 0 };
  Object.defineProperties(element, {
    clientHeight: { get: () => geometry.height },
    scrollHeight: { get: () => geometry.contentHeight },
    scrollTop: {
      get: () => geometry.top,
      set: (value: number) => {
        geometry.top = Math.max(0, Math.min(value, geometry.contentHeight - geometry.height));
      },
    },
  });
  return geometry;
};

const renderBlock = async (height = 180, contentHeight = 600) => {
  const view = render(<ThinkingBlock text="Reasoning" reasoningInProgress />);
  const scroller = view.container.querySelector<HTMLDivElement>(".overflow-y-auto")!;
  const geometry = measure(scroller, height, contentHeight);
  resize(scroller);
  await settleScrolling();
  const fade = (edge: "top" | "bottom") => {
    const overlay = view.container.querySelector<HTMLDivElement>(`[data-reasoning-fade="${edge}"]`)!;
    return { height: Number.parseFloat(overlay.style.height), opacity: Number.parseFloat(overlay.style.opacity) };
  };
  const scroll = (top: number) => {
    scroller.scrollTop = top;
    fireEvent.scroll(scroller);
  };
  return { view, scroller, geometry, fade, scroll };
};

it("keeps a subtle end fade for fully visible text without fading the top", async () => {
  const { fade } = await renderBlock(100, 100);
  expect(fade("top")).toEqual({ height: 0, opacity: 0 });
  expect(fade("bottom").height).toBe(21);
  expect(fade("bottom").opacity).toBeCloseTo(1 / 3);
});

it("ramps each clipped edge over 32 hidden pixels and saturates at 42 px and two-thirds opacity", async () => {
  const { fade, scroll } = await renderBlock();
  expect(fade("top").height).toBe(42);
  expect(fade("top").opacity).toBeCloseTo(2 / 3);
  expect(fade("bottom").height).toBe(21);
  expect(fade("bottom").opacity).toBeCloseTo(1 / 3);

  scroll(0);
  expect(fade("top")).toEqual({ height: 0, opacity: 0 });
  expect(fade("bottom").height).toBe(42);
  expect(fade("bottom").opacity).toBeCloseTo(2 / 3);
  scroll(16);
  expect(fade("top").height).toBe(21);
  expect(fade("top").opacity).toBeCloseTo(1 / 3);
  scroll(404);
  expect(fade("bottom")).toEqual({ height: 31.5, opacity: 0.5 });
});

it("follows appended reasoning only while pinned and refreshes fades when unpinned", async () => {
  const { view, scroller, geometry, fade, scroll } = await renderBlock();
  geometry.contentHeight = 700;
  view.rerender(<ThinkingBlock text="More reasoning" reasoningInProgress />);
  expect(scroller.scrollTop).toBe(520);
  expect(fade("bottom").height).toBe(21);
  await settleScrolling();

  scroll(496);
  expect(fade("bottom").height).toBe(36.75);
  geometry.contentHeight = 800;
  view.rerender(<ThinkingBlock text="Further reasoning" reasoningInProgress />);
  expect(scroller.scrollTop).toBe(496);
  expect(fade("bottom").height).toBe(42);
  expect(fade("bottom").opacity).toBeCloseTo(2 / 3);
});

it("tracks content and viewport resizes and caps each fade at half the viewport", async () => {
  const { scroller, geometry, fade, scroll } = await renderBlock();
  geometry.contentHeight = 700;
  resize(scroller.firstElementChild!);
  expect(scroller.scrollTop).toBe(520);
  expect(fade("bottom").height).toBe(21);
  await settleScrolling();

  scroll(100);
  geometry.height = 40;
  resize(scroller);
  expect(scroller.scrollTop).toBe(100);
  expect(fade("top").height).toBe(20);
  expect(fade("bottom").height).toBe(20);
});

it("recalculates fades after a completed block is reopened", async () => {
  const { view, scroll, fade } = await renderBlock();
  scroll(0);
  view.rerender(<ThinkingBlock text="Reasoning" reasoningInProgress={false} />);
  await waitFor(() => expect(view.container.querySelector(".overflow-y-auto")).toBeNull());
  fireEvent.click(screen.getByRole("button", { name: "Thinking" }));
  const scroller = view.container.querySelector<HTMLDivElement>(".overflow-y-auto")!;
  measure(scroller, 100, 240);
  resize(scroller);
  expect(fade("top")).toEqual({ height: 0, opacity: 0 });
  expect(fade("bottom").height).toBe(42);
  expect(fade("bottom").opacity).toBeCloseTo(2 / 3);
});
