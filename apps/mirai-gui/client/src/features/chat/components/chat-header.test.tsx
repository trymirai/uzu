import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useChatStore } from "@/stores/use-chat-store";
import SavedChats from "@/features/chat-history/components/saved-chats";
import { ChatHeader } from "./chat-header";
import { UNTITLED_CHAT_TITLE } from "@/constants/chat";

const route = vi.hoisted(() => ({ chatId: "first" as string | undefined }));
vi.mock("@tanstack/react-router", () => ({
  useParams: () => route,
  useNavigate: () => vi.fn(),
}));

const chatDefaults = useChatStore.getState();
const sidebarDefaults = useSidebarStore.getState();
const observers: Observer[] = [];

class Observer {
  observed: Element | undefined;
  disconnect = vi.fn();
  constructor(
    readonly callback: IntersectionObserverCallback,
    readonly options: IntersectionObserverInit,
  ) {
    observers.push(this);
  }
  observe(element: Element) {
    this.observed = element;
  }
  emit(ratio: number) {
    this.callback(
      [{ target: this.observed, isIntersecting: ratio > 0, intersectionRatio: ratio } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver,
    );
  }
}

const Harness = () => {
  const isOpen = useSidebarStore((state) => state.isOpen);
  return (
    <>
      <SavedChats />
      {route.chatId && <ChatHeader title={`Chat ${route.chatId}`} isSidebarOpen={isOpen} />}
    </>
  );
};

const titleWrapper = () => screen.getByRole("heading", { hidden: true }).parentElement!;

beforeEach(() => {
  observers.length = 0;
  route.chatId = "first";
  vi.stubGlobal("IntersectionObserver", Observer);
  useSidebarStore.setState({ ...sidebarDefaults, isOpen: true, chatVisibility: null }, true);
  useChatStore.setState(
    {
      ...chatDefaults,
      loadSavedChats: vi.fn(async () => {}),
      savedChats: ["first", "second"].map((id) => ({
        id,
        title: `Chat ${id}`,
        createdAt: 1,
        updatedAt: 1,
        messageCount: 2,
      })),
    },
    true,
  );
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useChatStore.setState(chatDefaults, true);
  useSidebarStore.setState(sidebarDefaults, true);
});

it("fades the title smoothly between 15% and 75% of the selected row being clipped", () => {
  render(<Harness />);
  const row = screen.getByRole("button", { name: "Chat first" });
  const observer = observers[0]!;
  expect(observer.observed).toBe(row);
  expect(observer.options.root).toBe(row.parentElement!.parentElement);
  expect(observer.options.threshold).toEqual(Array.from({ length: 101 }, (_, index) => index / 100));
  expect(observer.options.root).toHaveProperty("className", expect.stringContaining("overflow-y-auto"));
  const heading = screen.getByRole("heading");
  const wrapper = titleWrapper();
  expect(wrapper.style.opacity).toBe("1");

  act(() => observer.emit(1));
  expect(wrapper.style.opacity).toBe("0");
  expect(wrapper.getAttribute("aria-hidden")).toBe("true");
  expect(screen.getByRole("heading", { hidden: true })).toBe(heading);
  expect(wrapper.className).toContain("transition-[padding-left,opacity]");
  expect(wrapper.className).toContain("duration-300");
  expect(wrapper.className).toContain("motion-reduce:transition-none");

  for (const ratio of [0.9, 0.85]) {
    act(() => observer.emit(ratio));
    expect(wrapper.style.opacity).toBe("0");
    expect(wrapper.getAttribute("aria-hidden")).toBe("true");
  }
  for (const [ratio, opacity] of [
    [0.7, 0.15625],
    [0.55, 0.5],
    [0.4, 0.84375],
  ] as const) {
    act(() => observer.emit(ratio));
    expect(Number(wrapper.style.opacity)).toBeCloseTo(opacity);
  }
  expect(wrapper.getAttribute("aria-hidden")).toBe("false");
  act(() => observer.emit(0.25));
  expect(wrapper.style.opacity).toBe("1");
  act(() => observer.emit(0));
  expect(wrapper.style.opacity).toBe("1");
  act(() => observer.emit(1));
  expect(wrapper.style.opacity).toBe("0");
});

it("retains row visibility while closed so reopening starts the title fade with the sidebar", () => {
  render(<Harness />);
  const observer = observers[0]!;
  act(() => observer.emit(1));
  expect(titleWrapper().style.opacity).toBe("0");
  act(() => useSidebarStore.getState().toggle());
  expect(observer.disconnect).not.toHaveBeenCalled();
  expect(titleWrapper().style.opacity).toBe("1");
  expect(useSidebarStore.getState().chatVisibility).toEqual({ chatId: "first", ratio: 1 });

  act(() => useSidebarStore.getState().toggle());
  expect(observers).toHaveLength(1);
  expect(titleWrapper().style.opacity).toBe("0");

  act(() => useSidebarStore.getState().toggle());
  act(() => observer.emit(0.55));
  expect(titleWrapper().style.opacity).toBe("1");
  act(() => useSidebarStore.getState().toggle());
  expect(Number(titleWrapper().style.opacity)).toBeCloseTo(0.5);
});

it("tracks the new active row on chat switches and clears visibility when it disappears", () => {
  const view = render(<Harness />);
  const old = observers[0]!;
  act(() => old.emit(1));
  route.chatId = "second";
  view.rerender(<Harness />);
  expect(titleWrapper().style.opacity).toBe("1");
  expect(observers.at(-1)!.observed).toBe(screen.getByRole("button", { name: "Chat second" }));
  act(() => old.emit(1));
  expect(useSidebarStore.getState().chatVisibility).toEqual({ chatId: "second", ratio: 0 });
  act(() => observers.at(-1)!.emit(1));
  expect(titleWrapper().style.opacity).toBe("0");

  act(() => useChatStore.setState({ savedChats: [] }));
  expect(titleWrapper().style.opacity).toBe("1");
  expect(useSidebarStore.getState().chatVisibility).toEqual({ chatId: "second", ratio: 0 });
});

const measureRowAt = (top: number) => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return this.getAttribute("role") === "button" ? new DOMRect(0, top, 200, 20) : new DOMRect(0, 100, 200, 200);
  });
};

it.each(["Models", "Chats"])("does not flash a visible sidebar title when entering from %s", () => {
  measureRowAt(140);
  route.chatId = undefined;
  const view = render(<Harness />);
  expect(screen.queryByRole("heading", { hidden: true })).toBeNull();
  route.chatId = "first";
  view.rerender(<Harness />);

  // No IntersectionObserver callback has been delivered yet.
  expect(useSidebarStore.getState().chatVisibility).toEqual({ chatId: "first", ratio: 1 });
  expect(titleWrapper().style.opacity).toBe("0");
  act(() => observers[0]!.emit(1));
  expect(titleWrapper().style.opacity).toBe("0");
});

it("initializes clipped rows before the first observer callback", () => {
  measureRowAt(91);
  const view = render(<Harness />);
  expect(useSidebarStore.getState().chatVisibility).toEqual({ chatId: "first", ratio: 0.55 });
  expect(Number(titleWrapper().style.opacity)).toBeCloseTo(0.5);

  measureRowAt(400);
  route.chatId = "second";
  view.rerender(<Harness />);
  expect(titleWrapper().style.opacity).toBe("1");
});

it("waits for current-row visibility only while the sidebar is open", () => {
  const view = render(<ChatHeader title="Cities" isSidebarOpen />);
  expect(titleWrapper().style.opacity).toBe("0");
  act(() => useSidebarStore.getState().setChatVisibility({ chatId: "another", ratio: 0 }));
  expect(titleWrapper().style.opacity).toBe("0");
  view.rerender(<ChatHeader title="Cities" isSidebarOpen={false} />);
  expect(titleWrapper().style.opacity).toBe("1");
});

it("keeps the title visible when its row is not mounted or observation is unavailable", () => {
  route.chatId = "unsaved";
  const view = render(<Harness />);
  expect(observers).toHaveLength(0);
  expect(titleWrapper().style.opacity).toBe("1");
  vi.stubGlobal("IntersectionObserver", undefined);
  route.chatId = "first";
  view.rerender(<Harness />);
  expect(observers).toHaveLength(0);
  expect(titleWrapper().style.opacity).toBe("1");
});

it("keeps the new-chat placeholder hidden until the chat has a name", () => {
  const view = render(<ChatHeader title={UNTITLED_CHAT_TITLE} isSidebarOpen={false} />);
  expect(titleWrapper().getAttribute("aria-hidden")).toBe("true");
  expect(titleWrapper().style.opacity).toBe("0");
  view.rerender(<ChatHeader title="Cities" isSidebarOpen={false} />);
  expect(titleWrapper().getAttribute("aria-hidden")).toBe("false");
  expect(titleWrapper().style.opacity).toBe("1");
});
