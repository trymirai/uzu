import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { Button } from ".";

afterEach(cleanup);

it("restores focus after pointer interaction without showing a keyboard ring", () => {
  const onFocus = vi.fn();
  const ref = createRef<HTMLButtonElement>();
  render(
    <>
      <Button ref={ref} onFocus={onFocus}>
        Settings
      </Button>
      <button type="button">Close</button>
    </>,
  );
  const trigger = screen.getByRole("button", { name: "Settings" });
  const close = screen.getByRole("button", { name: "Close" });

  fireEvent.mouseDown(trigger);
  act(() => trigger.focus());
  fireEvent.mouseDown(close);
  act(() => close.focus());
  act(() => ref.current?.focus());

  expect(document.activeElement).toBe(trigger);
  expect(trigger.hasAttribute("data-focus")).toBe(false);
  expect(onFocus).toHaveBeenCalledTimes(2);
});

it("keeps keyboard-restored focus visible until pointer interaction", () => {
  render(
    <>
      <Button>Settings</Button>
      <button type="button">Close</button>
    </>,
  );
  const trigger = screen.getByRole("button", { name: "Settings" });
  const close = screen.getByRole("button", { name: "Close" });

  fireEvent.keyDown(document, { key: "Tab" });
  act(() => trigger.focus());
  act(() => close.focus());
  fireEvent.keyDown(close, { key: "Escape" });
  act(() => trigger.focus());

  expect(document.activeElement).toBe(trigger);
  expect(trigger.hasAttribute("data-focus")).toBe(true);
  fireEvent.mouseDown(trigger);
  expect(trigger.hasAttribute("data-focus")).toBe(false);
});

it("preserves anchor refs, keyboard focus, and disabled/loading behavior", () => {
  const ref = createRef<HTMLAnchorElement>();
  const onClick = vi.fn();
  const { rerender } = render(
    <Button href="/settings" ref={ref} onClick={onClick}>
      Settings
    </Button>,
  );
  const link = screen.getByRole("link", { name: "Settings" });

  fireEvent.keyDown(document, { key: "Tab" });
  act(() => ref.current?.focus());
  expect(document.activeElement).toBe(link);
  expect(link.hasAttribute("data-focus")).toBe(true);

  rerender(
    <Button href="/settings" ref={ref} onClick={onClick} loading>
      Settings
    </Button>,
  );
  expect(ref.current?.hasAttribute("href")).toBe(false);
  expect(ref.current?.tabIndex).toBe(-1);
  fireEvent.click(ref.current!);
  expect(onClick).not.toHaveBeenCalled();

  rerender(
    <Button disabled onClick={onClick}>
      Settings
    </Button>,
  );
  const button = screen.getByRole("button", { name: "Settings" }) as HTMLButtonElement;
  expect(button.disabled).toBe(true);
  fireEvent.click(button);
  expect(onClick).not.toHaveBeenCalled();
});
