import { beforeEach, expect, it, vi } from "vitest";
import type { SessionLoadingPayload, SessionStatePayload } from "@/platform/services/session";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { bindChatLoading, bindChatSessionState } from "./adapters";

const mocks = vi.hoisted(() => ({
  loading: null as ((p: SessionLoadingPayload) => void) | null,
  state: null as ((p: SessionStatePayload) => void) | null,
}));
vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({
    session: {
      onSessionLoading: (cb: (p: SessionLoadingPayload) => void) => {
        mocks.loading = cb;
        return () => {};
      },
      onSessionState: (cb: (p: SessionStatePayload) => void) => {
        mocks.state = cb;
        return () => {};
      },
    },
  }),
}));

const sessionDefaults = useChatSessionStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const REPO = "vendor/model";

beforeEach(() => {
  useChatSessionStore.setState(sessionDefaults, true);
  useRuntimeSessionStore.setState(runtimeDefaults, true);
  bindChatLoading();
  bindChatSessionState();
});

it("marks the model as loading on the start event", () => {
  mocks.loading?.({ status: "start", repoId: REPO });

  expect(useChatSessionStore.getState().isModelLoading).toBe(true);
  expect(useChatSessionStore.getState().operationState).toBe("loading");
  expect(useRuntimeSessionStore.getState().loadingSession).toEqual({ repoId: REPO });
});

it("clears the loading state on the error event", () => {
  mocks.loading?.({ status: "start", repoId: REPO });
  mocks.loading?.({ status: "error", repoId: REPO });

  expect(useChatSessionStore.getState().isModelLoading).toBe(false);
  expect(useChatSessionStore.getState().operationState).toBe("idle");
  expect(useRuntimeSessionStore.getState().loadingSession).toBeNull();
});

it("turns a finished load into the resident session", () => {
  mocks.loading?.({ status: "start", repoId: REPO });
  mocks.state?.({ active: true, repoId: REPO, isEjecting: false });

  expect(useChatSessionStore.getState().isModelLoading).toBe(false);
  expect(useRuntimeSessionStore.getState().loadingSession).toBeNull();
  expect(useRuntimeSessionStore.getState().residentSession).toEqual({ repoId: REPO });
});
