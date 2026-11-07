import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WindowControls } from "./window-controls";

vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({ system: { onWindowMaximizedChange: () => () => {} } }),
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("dims the caption glyphs while the window is inactive", () => {
  const focused = vi.spyOn(document, "hasFocus").mockReturnValue(true);
  render(<WindowControls />);
  const close = screen.getByLabelText("Close");
  expect(close.className).not.toContain("text-label-title/40");

  focused.mockReturnValue(false);
  act(() => void window.dispatchEvent(new FocusEvent("blur")));
  expect(close.className).toContain("text-label-title/40");

  focused.mockReturnValue(true);
  act(() => void window.dispatchEvent(new FocusEvent("focus")));
  expect(close.className).not.toContain("text-label-title/40");
});
