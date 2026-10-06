import { cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useChatStore } from "@/stores/use-chat-store";
import { useSaveFailureToast } from "./use-save-failure-toast";

const mocks = vi.hoisted(() => ({ error: vi.fn() }));
vi.mock("@/components/ui/toast/use-toast", () => ({ useToast: () => ({ error: mocks.error }) }));

afterEach(cleanup);

beforeEach(() => {
  mocks.error.mockClear();
  useChatStore.setState({ saveFailureCount: 0 });
});

it("does not warn about failures that happened before it mounted", () => {
  useChatStore.setState({ saveFailureCount: 3 });
  renderHook(() => useSaveFailureToast());

  expect(mocks.error).not.toHaveBeenCalled();
});
