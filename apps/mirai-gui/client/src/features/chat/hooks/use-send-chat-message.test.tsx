import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ChatNotFoundError, type ChatData, type ChatMetadata } from "@/platform/services/storage";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { Roles } from "@/types/chat";
import type { ToastApi } from "@/components/ui/toast/use-toast";
import type { StartStreamOptions } from "./use-llm-stream";
import { useSendChatMessage } from "./use-send-chat-message";
import { useStopChatRun } from "./use-stop-chat-run";
import type { ChatComposerState } from "./use-chat-composer-state";

const mocks = vi.hoisted(() => ({ platform: {} as Record<string, unknown> }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => mocks.platform }));

const CHAT_A = "a";
const CHAT_B = "b";
const chatDefaults = useChatStore.getState();
const sessionDefaults = useChatSessionStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const metadata: ChatMetadata = {
  id: CHAT_A,
  title: "Untitled",
  messageCount: 0,
  createdAt: 1,
  updatedAt: 1,
  modelId: "model",
  modelName: "Model",
};

const startStream = vi.fn<(options: StartStreamOptions) => Promise<void>>(async () => {
  expect(useChatSessionStore.getState().operationState).toBe("running");
});
const composer = { attachedFiles: [], clear: vi.fn() } as unknown as ChatComposerState;
const toast = { error: vi.fn(), info: vi.fn(), warning: vi.fn() } as unknown as ToastApi;
const noop = () => {};

const renderSend = (chatId = CHAT_A) =>
  renderHook(() => {
    const { stop } = useStopChatRun({
      chatId,
      cancelStream: async () => {},
    });
    const send = useSendChatMessage({
      chatId,
      composer,
      globalInstructions: "",
      runStatusBlockReason: null,
      toast,
      setScrollTargetId: noop,
      startStream,
    });
    return { send, stop };
  });

const generateTitle = vi.fn();
const cancelTitleGen = vi.fn();
const ejectSession = vi.fn(async () => {});
let chats = new Map<string, ChatData>();
const updateChatTitle = vi.fn(async (id: string, title: string, expectedTitle?: string) => {
  const chat = chats.get(id);
  if (!chat) return;
  if (expectedTitle !== undefined && chat.metadata.title !== expectedTitle) return;
  chat.metadata.title = title;
});

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.setState(chatDefaults, true);
  useChatSessionStore.setState(sessionDefaults, true);
  useRuntimeSessionStore.setState(runtimeDefaults, true);
  useChatStore.getState().createNewChat(CHAT_A);
  useChatStore.getState().setChatModel(CHAT_A, "model", "Model");
  chats = new Map<string, ChatData>();
  generateTitle.mockResolvedValue("Title");
  cancelTitleGen.mockResolvedValue(undefined);
  mocks.platform = {
    storage: {
      createOrReplaceChat: vi.fn(async (chat: ChatData) => {
        chats.set(chat.metadata.id, structuredClone(chat));
      }),
      appendMessage: vi.fn(async (id: string, message: ChatData["messages"][number]) => {
        const chat = chats.get(id);
        if (!chat) throw new ChatNotFoundError(id);
        chat.messages = [...chat.messages.filter((m) => m.id !== message.id), structuredClone(message)];
      }),
      loadChat: vi.fn(async (id: string) => structuredClone(chats.get(id) ?? null)),
      listChats: vi.fn(async () => [...chats.values()].map((c) => c.metadata)),
      updateChatTitle,
      updateStoredMessage: vi.fn(async () => {}),
      removeMessage: vi.fn(async (id: string, messageId: string) => {
        const chat = chats.get(id);
        if (chat) chat.messages = chat.messages.filter((m) => m.id !== messageId);
      }),
      deleteChat: vi.fn(async (id: string) => {
        chats.delete(id);
      }),
    },
    chat: { generateTitle, cancelTitleGen },
    session: { ejectSession },
  };
});

afterEach(cleanup);

it("does not read another chat's history when the user navigates during persistence", async () => {
  let finishPersist!: () => void;
  useChatStore.setState({
    persistMessage: vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishPersist = resolve;
          }),
      )
      .mockResolvedValue(undefined),
    generateChatTitle: vi.fn(async () => ({ ok: true })),
  });
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("question in a");
  });
  await waitFor(() => expect(finishPersist).toBeTypeOf("function"));

  useChatStore.getState().createNewChat(CHAT_B);
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.User, text: "private content of b" });
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.Assistant, text: "answer in b" });
  await act(async () => {
    finishPersist();
    await sending;
  });

  expect(JSON.stringify(startStream.mock.calls)).not.toContain("private content of b");
  expect(useChatStore.getState().messages).toHaveLength(2);
});

it("does not start a response after Stop during title generation following a remount", async () => {
  useChatStore.setState({ savedChats: [metadata] });
  let resolveTitle!: (value: string) => void;
  generateTitle.mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveTitle = resolve;
      }),
  );
  cancelTitleGen.mockImplementation(async () => {
    resolveTitle("");
  });
  const original = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = original.result.current.send("hello");
  });
  await waitFor(() => expect(generateTitle).toHaveBeenCalled());
  original.unmount();

  const remounted = renderSend();
  await act(async () => {
    await remounted.result.current.stop();
    await sending;
  });

  expect(cancelTitleGen).toHaveBeenCalledTimes(1);
  expect(startStream).not.toHaveBeenCalled();
  expect(updateChatTitle).not.toHaveBeenCalled();
});

it("keeps the Stop of one chat's title generation when another chat sends meanwhile", async () => {
  useChatStore.setState({ savedChats: [metadata] });
  let resolveTitle!: (value: string) => void;
  generateTitle.mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveTitle = resolve;
      }),
  );
  const a = renderSend(CHAT_A);
  let sending!: Promise<void>;
  act(() => {
    sending = a.result.current.send("hello");
  });
  await waitFor(() => expect(generateTitle).toHaveBeenCalled());
  await act(() => a.result.current.stop());

  useChatStore.getState().createNewChat(CHAT_B);
  useChatStore.getState().setChatModel(CHAT_B, "model", "Model");
  const b = renderSend(CHAT_B);
  await act(async () => void (await b.result.current.send("meanwhile")));
  expect(toast.error).toHaveBeenCalledWith("Generating chat title, please wait");

  await act(async () => {
    resolveTitle("");
    await sending;
  });

  expect(startStream).not.toHaveBeenCalled();
  expect(updateChatTitle).not.toHaveBeenCalled();
});

it("keeps another chat's history out of a send that waited for a model eject", async () => {
  useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.User, text: "history A" });
  const answerId = useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.Assistant, text: "answer A" }).id;
  await useChatStore.getState().persistMessage(CHAT_A, answerId);
  useRuntimeSessionStore.setState({ residentSession: { repoId: "old-model" } });
  const eject = deferred<void>();
  ejectSession.mockReturnValueOnce(eject.promise);
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("prompt A");
  });
  await waitFor(() => expect(ejectSession).toHaveBeenCalled());

  useChatStore.getState().createNewChat(CHAT_B);
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.User, text: "private history B" });
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.Assistant, text: "private answer B" });
  await act(async () => {
    useRuntimeSessionStore.setState({ residentSession: null });
    eject.resolve();
    await sending;
  });

  expect(startStream).toHaveBeenCalledTimes(1);
  const sent = startStream.mock.calls[0]?.[0].messages.map((m) => m.content).join("\n");
  expect(sent).toContain("history A");
  expect(sent).not.toContain("private history B");
  expect(useChatStore.getState().messages.map((m) => m.text)).toEqual(["private history B", "private answer B"]);
  expect(chats.get(CHAT_A)?.messages.map((m) => m.text)).toContain("prompt A");
});

it("keeps a failed save of one chat out of the chat opened meanwhile", async () => {
  const save = deferred<void>();
  (mocks.platform.storage as { createOrReplaceChat: ReturnType<typeof vi.fn> }).createOrReplaceChat.mockReturnValueOnce(
    save.promise,
  );
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("prompt A");
  });
  await waitFor(() => expect(useChatStore.getState().savedChats.map((c) => c.id)).toContain(CHAT_A));

  useChatStore.getState().createNewChat(CHAT_B);
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.User, text: "private history B" });
  await act(async () => {
    save.reject(new Error("disk full"));
    await sending;
  });

  expect(useChatStore.getState().messages.map((m) => m.text)).toEqual(["private history B"]);
  expect(chats.get(CHAT_A)?.messages.map((m) => m.error)).toEqual(["Error: disk full"]);
});

it("does not start a reply for a chat deleted during title generation", async () => {
  const title = deferred<string>();
  generateTitle.mockReturnValueOnce(title.promise);
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("prompt A");
  });
  await waitFor(() => expect(generateTitle).toHaveBeenCalled());

  await act(() => useChatStore.getState().deleteChat(CHAT_A));
  await act(async () => {
    title.resolve("");
    await sending;
  });

  expect(cancelTitleGen).toHaveBeenCalledTimes(1);
  expect(startStream).not.toHaveBeenCalled();
  expect(chats.has(CHAT_A)).toBe(false);
});

it("keeps a manual rename made while the title was generating", async () => {
  const title = deferred<string>();
  generateTitle.mockReturnValueOnce(title.promise);
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("prompt A");
  });
  await waitFor(() => expect(generateTitle).toHaveBeenCalled());

  await act(() => useChatStore.getState().updateChatTitle(CHAT_A, "Chosen by user"));
  await act(async () => {
    title.resolve("Automatic title");
    await sending;
  });

  expect(chats.get(CHAT_A)?.metadata.title).toBe("Chosen by user");
  expect(startStream).toHaveBeenCalledTimes(1);
});

it("reports a failed eject instead of rejecting the send", async () => {
  useRuntimeSessionStore.setState({ residentSession: { repoId: "old-model" } });
  ejectSession.mockRejectedValueOnce(new Error("IPC unavailable"));
  vi.spyOn(console, "error").mockImplementation(() => {});
  const hook = renderSend();

  await act(async () => void (await hook.result.current.send("prompt A")));

  expect(toast.error).toHaveBeenCalledWith("Failed to eject previous model");
  expect(startStream).not.toHaveBeenCalled();
  expect(composer.clear).not.toHaveBeenCalled();
});

it("clears the composer with the sent text once the message is accepted", async () => {
  const hook = renderSend();

  await act(async () => void (await hook.result.current.send("prompt A")));

  expect(composer.clear).toHaveBeenCalledWith("prompt A");
  await waitFor(() => expect(startStream).toHaveBeenCalled());
});

it("stores a new chat's first exchange without duplicates", async () => {
  const hook = renderSend();

  await act(() => hook.result.current.send("hello"));

  const stored = chats.get(CHAT_A)?.messages ?? [];
  expect(stored.map((m) => m.sender)).toEqual([Roles.User, Roles.Assistant]);
  expect(new Set(stored.map((m) => m.id)).size).toBe(2);
});

it("refuses a second send while the first is still being written", async () => {
  const firstWrite = deferred<void>();
  (mocks.platform.storage as { createOrReplaceChat: ReturnType<typeof vi.fn> }).createOrReplaceChat.mockReturnValueOnce(
    firstWrite.promise,
  );
  const hook = renderSend();

  let first!: Promise<void>;
  act(() => {
    first = hook.result.current.send("first");
  });
  await waitFor(() => expect(useChatStore.getState().messages).toHaveLength(2));
  await act(() => hook.result.current.send("second"));

  expect(toast.error).toHaveBeenCalledWith("A response is already generating.");
  expect(useChatStore.getState().messages.map((m) => m.text)).toEqual(["first", ""]);

  await act(async () => {
    firstWrite.resolve();
    await first;
  });
  expect(startStream).toHaveBeenCalledTimes(1);
  expect(useChatSessionStore.getState().operationState).toBe("idle");
});

it("turns the placeholder into the error bubble when the first write fails", async () => {
  (mocks.platform.storage as { createOrReplaceChat: ReturnType<typeof vi.fn> }).createOrReplaceChat
    .mockRejectedValueOnce(new Error("disk full"))
    .mockImplementation(async (chat: ChatData) => {
      chats.set(chat.metadata.id, structuredClone(chat));
    });
  const hook = renderSend();

  await act(() => hook.result.current.send("hello"));

  const messages = useChatStore.getState().messages;
  expect(messages.map((m) => [m.sender, m.error ?? m.text])).toEqual([
    [Roles.User, "hello"],
    [Roles.Assistant, "Error: disk full"],
  ]);
  expect(chats.get(CHAT_A)?.messages.map((m) => m.id)).toEqual(messages.map((m) => m.id));
  expect(useChatSessionStore.getState()).toMatchObject({ loadingMessage: null, operationState: "idle" });
});
