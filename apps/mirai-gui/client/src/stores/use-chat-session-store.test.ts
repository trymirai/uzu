import { beforeEach, expect, it, vi } from "vitest";
import { useChatSessionStore } from "./use-chat-session-store";

const platform = vi.hoisted(() => ({ chat: { cancelTitleGen: vi.fn(), cancelRun: vi.fn() } }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => platform }));

const defaults = useChatSessionStore.getState();

beforeEach(() => {
  vi.resetAllMocks();
  useChatSessionStore.setState(defaults, true);
});

const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => (resolve = done));
  return { promise, resolve };
};

it("ignores a loading-message clear that comes from another chat", () => {
  const { setLoadingMessage } = useChatSessionStore.getState();
  setLoadingMessage("a", "m1");

  setLoadingMessage("b", null);
  expect(useChatSessionStore.getState().loadingMessage).toEqual({ chatId: "a", messageId: "m1" });

  setLoadingMessage("a", null);
  expect(useChatSessionStore.getState().loadingMessage).toBeNull();
});

it("marks only the stopped chat's message as canceled", () => {
  const { setCanceledMessage } = useChatSessionStore.getState();
  setCanceledMessage("a", "m1");

  setCanceledMessage("b", null);
  expect(useChatSessionStore.getState().canceledMessage).toEqual({ chatId: "a", messageId: "m1" });
});

it("aborts matching chat preparation synchronously before awaiting backend cancellation", async () => {
  const preparing = deferred();
  const titleCancel = deferred();
  let signal!: AbortSignal;
  const session = useChatSessionStore.getState();
  const operation = session.withOperation(
    "running",
    async (current) => {
      signal = current;
      await preparing.promise;
    },
    "a",
  );
  await session.cancelActiveRunForChat("b");
  expect(signal.aborted).toBe(false);

  session.setTitleGenChat("a");
  platform.chat.cancelTitleGen.mockReturnValueOnce(titleCancel.promise);
  const canceled = session.cancelActiveRunForChat("a");
  expect(signal.aborted).toBe(true);
  expect(platform.chat.cancelTitleGen).toHaveBeenCalledOnce();
  expect(useChatSessionStore.getState().operationState).toBe("running");
  titleCancel.resolve();
  await canceled;
  preparing.resolve();
  await operation;
  expect(useChatSessionStore.getState().operationState).toBe("idle");
});

it("keeps a running chat cancelable after an overlapping stop operation finishes", async () => {
  const preparing = deferred();
  let runningSignal!: AbortSignal;
  let stoppingSignal!: AbortSignal;
  const session = useChatSessionStore.getState();
  const operation = session.withOperation(
    "running",
    async (signal) => {
      runningSignal = signal;
      await preparing.promise;
    },
    "a",
  );
  await session.withOperation(
    "stopping",
    async (signal) => {
      stoppingSignal = signal;
    },
    "a",
  );
  await session.cancelActiveRunForChat("a");
  expect(runningSignal.aborted).toBe(true);
  expect(stoppingSignal.aborted).toBe(false);
  preparing.resolve();
  await operation;
});

it("an earlier operation's cleanup leaves the overlapping operation and its cancellation intact", async () => {
  const running = deferred();
  const stopping = deferred();
  let runningSignal!: AbortSignal;
  let stoppingSignal!: AbortSignal;
  const session = useChatSessionStore.getState();
  const first = session.withOperation(
    "running",
    async (signal) => {
      runningSignal = signal;
      await running.promise;
    },
    "a",
  );
  const second = session.withOperation(
    "stopping",
    async (signal) => {
      stoppingSignal = signal;
      await stopping.promise;
    },
    "b",
  );
  const stoppingId = useChatSessionStore.getState().operationId;
  running.resolve();
  await first;
  expect(useChatSessionStore.getState().operationState).toBe("stopping");
  expect(useChatSessionStore.getState().operationId).toBe(stoppingId);
  await session.cancelActiveRunForChat("a");
  expect(runningSignal.aborted).toBe(false);
  await session.cancelActiveRunForChat("b");
  expect(stoppingSignal.aborted).toBe(true);
  stopping.resolve();
  await second;
  expect(useChatSessionStore.getState().operationState).toBe("idle");
});

it("removes cancellation ownership when an operation fails", async () => {
  let signal!: AbortSignal;
  const session = useChatSessionStore.getState();
  await expect(
    session.withOperation(
      "running",
      async (current) => {
        signal = current;
        throw new Error("write failed");
      },
      "a",
    ),
  ).rejects.toThrow("write failed");
  await session.cancelActiveRunForChat("a");
  expect(signal.aborted).toBe(false);
  expect(useChatSessionStore.getState().operationState).toBe("idle");
});
