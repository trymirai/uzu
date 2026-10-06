import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { useChatComposerState } from "./use-chat-composer-state";

it("clears the draft that was sent", () => {
  const hook = renderHook(() => useChatComposerState());
  act(() => hook.result.current.setDraft("hello "));

  act(() => hook.result.current.clear("hello"));

  expect(hook.result.current.draft).toBe("");
});

it("keeps a draft typed while the send was in flight", () => {
  const hook = renderHook(() => useChatComposerState());
  act(() => hook.result.current.setDraft("first"));
  act(() => hook.result.current.setDraft("second"));

  act(() => hook.result.current.clear("first"));

  expect(hook.result.current.draft).toBe("second");
});
