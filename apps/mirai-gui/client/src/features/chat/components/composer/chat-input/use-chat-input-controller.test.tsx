import { act, renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { useChatInputController } from "./use-chat-input-controller";

const enter = (
  overrides: Partial<ReactKeyboardEvent<HTMLTextAreaElement>> & { isComposing?: boolean; keyCode?: number } = {},
) =>
  ({
    key: "Enter",
    shiftKey: false,
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    preventDefault: vi.fn(),
    nativeEvent: { isComposing: overrides.isComposing ?? false, keyCode: overrides.keyCode ?? 13 },
    ...overrides,
  }) as unknown as ReactKeyboardEvent<HTMLTextAreaElement>;

it("does not send on the Enter that confirms an IME composition", () => {
  const onSend = vi.fn();
  const hook = renderHook(() => useChatInputController({ value: "日本語", onSend }));

  const composing = enter({ isComposing: true });
  act(() => hook.result.current.handleKeyDown(composing));
  expect(onSend).not.toHaveBeenCalled();
  expect(composing.preventDefault).not.toHaveBeenCalled();

  const plain = enter();
  act(() => hook.result.current.handleKeyDown(plain));
  expect(onSend).toHaveBeenCalledTimes(1);
  expect(plain.preventDefault).toHaveBeenCalled();
});

it("does not send on the WebKit Enter that reports no composition but keyCode 229", () => {
  const onSend = vi.fn();
  const hook = renderHook(() => useChatInputController({ value: "日本語", onSend }));

  const webkitConfirm = enter({ keyCode: 229 });
  act(() => hook.result.current.handleKeyDown(webkitConfirm));

  expect(onSend).not.toHaveBeenCalled();
  expect(webkitConfirm.preventDefault).not.toHaveBeenCalled();
});
