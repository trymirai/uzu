import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ChatNotFoundError, type ChatData, type ChatMetadata } from "@/platform/services/storage";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { ModelKind } from "@/types/models";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { Roles } from "@/types/chat";
import type { ToastApi } from "@/components/ui/toast/use-toast";
import type { StartStreamOptions } from "./use-llm-stream";
import type { LlmRunResult } from "@/types/llm-stream";
import { emptyStats } from "@/platform/services/chat/empty-stats";
import { useSendChatMessage } from "./use-send-chat-message";
import { useStopChatRun } from "./use-stop-chat-run";
import type { ChatComposerState } from "./use-chat-composer-state";
import { attachmentStorage } from "../services/attachment-storage";

const mocks = vi.hoisted(() => ({ platform: {} as Record<string, unknown> }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => mocks.platform }));

const CHAT_A = "a";
const CHAT_B = "b";
const chatDefaults = useChatStore.getState();
const sessionDefaults = useChatSessionStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const paramsDefaults = useModelParamsStore.getState();
const metadata: ChatMetadata = {
  id: CHAT_A,
  title: "Untitled",
  messageCount: 0,
  createdAt: 1,
  updatedAt: 1,
  modelId: "model",
  modelName: "Model",
};

const startStream = vi.fn<(options: StartStreamOptions) => Promise<LlmRunResult | void>>(async () => {
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
const getModelChatNamingEnabled = vi.fn(async () => false);
let chats = new Map<string, ChatData>();
const editUserMessage = vi.fn(async (id: string, messageId: string, text: string) => {
  const chat = structuredClone(chats.get(id));
  if (!chat) throw new ChatNotFoundError(id);
  const index = chat.messages.findIndex((m) => m.id === messageId && m.sender === Roles.User);
  if (index < 0) throw new Error("User message not found");
  chat.messages = chat.messages.slice(0, index + 1);
  chat.messages[index] = { ...chat.messages[index]!, text };
  chat.metadata.messageCount = chat.messages.length;
  chats.set(id, chat);
  return structuredClone(chat);
});
const updateChatTitle = vi.fn(async (id: string, title: string, expectedTitle?: string) => {
  const chat = chats.get(id);
  if (!chat) return false;
  if (expectedTitle !== undefined && chat.metadata.title !== expectedTitle) return false;
  chat.metadata.title = title;
  return true;
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
  useModelParamsStore.setState(paramsDefaults, true);
  useModelsStore.setState({ models: [] });
  useChatStore.getState().createNewChat(CHAT_A);
  useChatStore.getState().setChatModel(CHAT_A, "model", "Model");
  chats = new Map<string, ChatData>();
  generateTitle.mockResolvedValue("Title");
  getModelChatNamingEnabled.mockResolvedValue(false);
  composer.attachedFiles = [];
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
      updateStoredMessage: vi.fn(
        async (id: string, messageId: string, patch: Partial<ChatData["messages"][number]>) => {
          const chat = chats.get(id);
          if (chat)
            chat.messages = chat.messages.map((m) => (m.id === messageId ? { ...m, ...structuredClone(patch) } : m));
        },
      ),
      editUserMessage,
      removeMessage: vi.fn(async (id: string, messageId: string) => {
        const chat = chats.get(id);
        if (chat) chat.messages = chat.messages.filter((m) => m.id !== messageId);
      }),
      deleteChat: vi.fn(async (id: string) => {
        chats.delete(id);
      }),
    },
    chat: { generateTitle, cancelTitleGen },
    settings: { getModelChatNamingEnabled },
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

it("does not launch fallback naming after Stop while loading the existing title", async () => {
  const loaded = deferred<ChatData | null>();
  const storage = mocks.platform.storage as { loadChat: ReturnType<typeof vi.fn> };
  storage.loadChat.mockReturnValueOnce(loaded.promise);
  const naming = useChatStore.getState().generateChatTitle(CHAT_A, "hello");
  await waitFor(() => expect(storage.loadChat).toHaveBeenCalled());
  await useChatSessionStore.getState().cancelActiveRunForChat(CHAT_A);
  loaded.resolve({ metadata, messages: [] });
  await naming;
  expect(generateTitle).not.toHaveBeenCalled();
  expect(updateChatTitle).not.toHaveBeenCalled();
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

it("skips separate title generation when the model named the first turn", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  startStream.mockResolvedValueOnce({ text: "Answer", stats: emptyStats(), finishReason: "Stop", chatName: "A name" });
  const hook = renderSend();

  await act(() => hook.result.current.send("question"));

  expect(startStream.mock.calls[0]?.[0].modelChatNamingEnabled).toBe(true);
  expect(startStream.mock.calls[0]?.[0].dateTimeToolEnabled).toBe(true);
  expect(startStream.mock.calls[0]?.[0].chartToolEnabled).toBe(true);
  expect(generateTitle).not.toHaveBeenCalled();
});

it.each([false, true])("sends a small model with tools enabled only after opt-in=%s", async (enabled) => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  useModelsStore.setState({
    models: [
      {
        repoId: "model",
        name: "Model",
        vendor: "Vendor",
        kind: ModelKind.Text,
        reasoning: { kind: "unsupported" },
        supportsTools: true,
        paramSize: 1_000_000_000,
      },
    ],
  });
  if (enabled)
    useModelParamsStore.setState({
      paramsByRepoId: {
        model: {
          sampling: { type: "Default" },
          modelChatNamingEnabled: true,
          dateTimeToolEnabled: true,
          chartToolEnabled: true,
        },
      },
    });
  const hook = renderSend();
  await act(() => hook.result.current.send("question"));
  expect(startStream).toHaveBeenCalledWith(
    expect.objectContaining({
      modelChatNamingEnabled: enabled,
      dateTimeToolEnabled: enabled,
      chartToolEnabled: enabled,
    }),
  );
});

it("ignores an old per-model naming opt-in while the global feature is off", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: {
      model: {
        sampling: { type: "Default" },
        modelChatNamingEnabled: true,
        dateTimeToolEnabled: false,
        chartToolEnabled: false,
      },
    },
  });
  startStream.mockResolvedValueOnce({ text: "Answer", stats: emptyStats(), finishReason: "Stop", chatName: "A name" });
  const hook = renderSend();
  await act(() => hook.result.current.send("question"));
  expect(startStream.mock.calls[0]?.[0]).toMatchObject({
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
  });
  expect(generateTitle).toHaveBeenCalledTimes(1);
});

it("uses separate title generation when the per-model naming tool is turned off", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  useModelParamsStore.setState({
    paramsByRepoId: { model: { sampling: { type: "Default" }, modelChatNamingEnabled: false } },
  });
  const hook = renderSend();
  await act(() => hook.result.current.send("question"));
  expect(startStream.mock.calls[0]?.[0]).toMatchObject({ modelChatNamingEnabled: false, dateTimeToolEnabled: true });
  expect(generateTitle).toHaveBeenCalledTimes(1);
});

it("waits for the first reply before falling back when no model name was provided", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  const reply = deferred<LlmRunResult>();
  startStream.mockReturnValueOnce(reply.promise);
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("question");
  });
  await waitFor(() => expect(startStream).toHaveBeenCalled());
  expect(generateTitle).not.toHaveBeenCalled();

  await act(async () => {
    reply.resolve({ text: "Answer", stats: emptyStats(), finishReason: "Stop" });
    await sending;
  });

  expect(generateTitle).toHaveBeenCalledTimes(1);
  expect(chats.get(CHAT_A)?.metadata.title).toBe("Title");
});

it("does not retry the fallback on later turns of an untitled chat", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.User, text: "earlier" });
  useChatStore.getState().addMessageTo(CHAT_A, { sender: Roles.Assistant, text: "earlier answer" });
  startStream.mockResolvedValueOnce({ text: "Answer", stats: emptyStats(), finishReason: "Stop" });
  const hook = renderSend();

  await act(() => hook.result.current.send("next question"));

  expect(generateTitle).not.toHaveBeenCalled();
});

it.each(["Cancelled", "Length", "ContextLimitReached"] as const)(
  "does not run fallback after %s",
  async (finishReason) => {
    getModelChatNamingEnabled.mockResolvedValue(true);
    startStream.mockResolvedValueOnce({ text: "Partial answer", stats: emptyStats(), finishReason });
    const hook = renderSend();

    await act(() => hook.result.current.send("question"));

    expect(generateTitle).not.toHaveBeenCalled();
  },
);

it("keeps a completed reply when fallback title generation fails", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  startStream.mockImplementationOnce(async (options) => {
    options.updateText(options.messageId, "Completed answer");
    return { text: "Completed answer", stats: emptyStats(), finishReason: "Stop" };
  });
  generateTitle.mockRejectedValueOnce(new Error("title failed"));
  const hook = renderSend();

  await act(() => hook.result.current.send("question"));

  expect(useChatStore.getState().messages.at(-1)).toMatchObject({ text: "Completed answer" });
  expect(useChatStore.getState().messages.at(-1)?.error).toBeUndefined();
  expect(toast.error).toHaveBeenCalledWith("Could not name the chat: title failed");
});

it("keeps the reply when Stop cancels the fallback title request", async () => {
  getModelChatNamingEnabled.mockResolvedValue(true);
  startStream.mockImplementationOnce(async (options) => {
    options.updateText(options.messageId, "Completed answer");
    return { text: "Completed answer", stats: emptyStats(), finishReason: "Stop" };
  });
  const title = deferred<string>();
  generateTitle.mockReturnValueOnce(title.promise);
  cancelTitleGen.mockImplementationOnce(async () => {
    title.resolve("");
  });
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("question");
  });
  await waitFor(() => expect(generateTitle).toHaveBeenCalled());

  await act(async () => {
    await hook.result.current.stop();
    await sending;
  });

  expect(useChatStore.getState().messages.at(-1)?.text).toBe("Completed answer");
  expect(chats.get(CHAT_A)?.messages).toHaveLength(2);
  expect(useChatSessionStore.getState().titleGenAbortChatId).toBeNull();
  expect(updateChatTitle).not.toHaveBeenCalled();
});

const seedEditableChat = () => {
  const messages: ChatData["messages"] = [
    { id: "u1", sender: Roles.User, text: "first question", timestamp: 1 },
    { id: "a1", sender: Roles.Assistant, text: "first answer", timestamp: 2 },
    { id: "u2", sender: Roles.User, text: "old question", timestamp: 3, attachmentIds: ["original-file"] },
    { id: "a2", sender: Roles.Assistant, text: "discarded answer", timestamp: 4 },
    { id: "u3", sender: Roles.User, text: "discarded follow-up", timestamp: 5 },
  ];
  const data = { metadata: { ...metadata, title: "Chosen title", messageCount: messages.length }, messages };
  chats.set(CHAT_A, structuredClone(data));
  useChatStore.setState({ messages, savedChats: [data.metadata] });
  attachmentStorage.saveFile({
    id: "original-file",
    name: "notes.txt",
    mimeType: "text/plain",
    size: 16,
    extension: "txt",
    content: "original notes",
  });
  return data;
};

it("replaces an earlier prompt and regenerates with only its retained history and attachments", async () => {
  seedEditableChat();
  composer.attachedFiles = [
    { id: "draft-file", name: "draft.txt", mimeType: "text/plain", size: 5, extension: "txt", content: "unsent draft" },
  ];
  const onSaved = vi.fn();
  const hook = renderSend();

  await act(() => hook.result.current.send("edited question", { messageId: "u2", onSaved }));

  expect(startStream).toHaveBeenCalledTimes(1);
  expect(startStream.mock.calls[0]?.[0].messages).toEqual([
    { role: Roles.User, content: "first question" },
    { role: Roles.Assistant, content: "first answer" },
    { role: Roles.User, content: "edited question\n\n```txt\noriginal notes\n```" },
  ]);
  const stored = chats.get(CHAT_A)!;
  expect(stored.messages.map((m) => m.text)).toEqual(["first question", "first answer", "edited question", ""]);
  expect(stored.messages[2]).toMatchObject({ id: "u2", timestamp: 3, attachmentIds: ["original-file"] });
  expect(stored.messages[3]?.versions).toBeUndefined();
  expect(stored.metadata.title).toBe("Chosen title");
  expect(useChatStore.getState().messages).toEqual(stored.messages);
  expect(composer.clear).not.toHaveBeenCalled();
  expect(composer.attachedFiles[0]?.id).toBe("draft-file");
  expect(onSaved).toHaveBeenCalledOnce();
});

it("leaves the entire conversation intact and does not generate when saving an edit fails", async () => {
  const original = seedEditableChat();
  editUserMessage.mockRejectedValueOnce(new Error("disk full"));
  const onSaved = vi.fn();
  const hook = renderSend();

  await act(async () => {
    await expect(hook.result.current.send("edited", { messageId: "u2", onSaved })).rejects.toThrow("disk full");
  });

  expect(chats.get(CHAT_A)).toEqual(original);
  expect(useChatStore.getState().messages).toEqual(original.messages);
  expect(useChatSessionStore.getState().operationState).toBe("idle");
  expect(startStream).not.toHaveBeenCalled();
  expect(onSaved).not.toHaveBeenCalled();
});

it("refuses edits until the previous operation, including final persistence, finishes", async () => {
  const original = seedEditableChat();
  useChatSessionStore.setState({ operationState: "running", isGenerating: false });
  const hook = renderSend();
  await act(() => hook.result.current.send("edited", { messageId: "u2", onSaved: noop }));

  expect(editUserMessage).not.toHaveBeenCalled();
  expect(chats.get(CHAT_A)).toEqual(original);
  expect(startStream).not.toHaveBeenCalled();
  expect(toast.error).toHaveBeenCalledWith("A response is already generating.");
});

it("closes the editor only after the edit reaches storage, before the new reply finishes", async () => {
  const original = seedEditableChat();
  const save = deferred<ChatData>();
  const response = deferred<void>();
  editUserMessage.mockReturnValueOnce(save.promise);
  startStream.mockReturnValueOnce(response.promise);
  const onSaved = vi.fn();
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("edited", { messageId: "u2", onSaved });
  });
  await waitFor(() => expect(editUserMessage).toHaveBeenCalled());
  expect(onSaved).not.toHaveBeenCalled();
  expect(useChatStore.getState().messages).toEqual(original.messages);

  const edited = structuredClone(original);
  edited.messages = [...edited.messages.slice(0, 2), { ...edited.messages[2]!, text: "edited" }];
  await act(async () => {
    chats.set(CHAT_A, edited);
    save.resolve(structuredClone(edited));
  });
  await waitFor(() => expect(startStream).toHaveBeenCalled());
  expect(onSaved).toHaveBeenCalledOnce();
  expect(useChatSessionStore.getState().operationState).toBe("running");
  await act(async () => {
    response.resolve();
    await sending;
  });
});

it("keeps edits and their replacement reply in the original chat when navigating during the save", async () => {
  const original = seedEditableChat();
  const save = deferred<ChatData>();
  editUserMessage.mockReturnValueOnce(save.promise);
  const hook = renderSend();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("edited", { messageId: "u2", onSaved: noop });
  });
  await waitFor(() => expect(editUserMessage).toHaveBeenCalled());
  useChatStore.getState().createNewChat(CHAT_B);
  useChatStore.getState().addMessageTo(CHAT_B, { sender: Roles.User, text: "private history B" });
  const edited = structuredClone(original);
  edited.messages = [...edited.messages.slice(0, 2), { ...edited.messages[2]!, text: "edited" }];
  await act(async () => {
    chats.set(CHAT_A, edited);
    save.resolve(structuredClone(edited));
    await sending;
  });

  expect(useChatStore.getState().messages.map((m) => m.text)).toEqual(["private history B"]);
  expect(chats.get(CHAT_A)?.messages.map((m) => m.text)).toEqual(["first question", "first answer", "edited", ""]);
  expect(JSON.stringify(startStream.mock.calls)).not.toContain("private history B");
  expect(startStream.mock.calls[0]?.[0].chatId).toBe(CHAT_A);
});

it("does not recreate a chat deleted while its edit is being saved", async () => {
  const original = seedEditableChat();
  const save = deferred<ChatData>();
  editUserMessage.mockReturnValueOnce(save.promise);
  const hook = renderSend();
  const onSaved = vi.fn();
  let sending!: Promise<void>;
  act(() => {
    sending = hook.result.current.send("edited", { messageId: "u2", onSaved });
  });
  await waitFor(() => expect(editUserMessage).toHaveBeenCalled());
  await act(() => useChatStore.getState().deleteChat(CHAT_A));
  await act(async () => {
    save.resolve(original);
    await sending;
  });

  expect(chats.has(CHAT_A)).toBe(false);
  expect(useChatStore.getState().savedChats).toEqual([]);
  expect(useChatStore.getState().messages).toEqual([]);
  expect(startStream).not.toHaveBeenCalled();
  expect(onSaved).not.toHaveBeenCalled();
  expect(useChatSessionStore.getState().operationState).toBe("idle");
});

it("retains the selected earlier response version when editing a later prompt", async () => {
  const data = seedEditableChat();
  const answer = data.messages[1]!;
  answer.versions = [
    { id: "v0", text: "first answer", modelId: "model", modelName: "Model", timestamp: 2 },
    { id: "v1", text: "chosen answer", modelId: "model", modelName: "Model", timestamp: 3 },
  ];
  answer.currentVersionIndex = 0;
  chats.set(CHAT_A, structuredClone(data));
  useChatStore.setState({ messages: structuredClone(data.messages) });
  useChatStore.getState().switchMessageVersion("a1", 1);
  const hook = renderSend();

  await act(() => hook.result.current.send("edited", { messageId: "u2", onSaved: noop }));

  expect(startStream.mock.calls[0]?.[0].messages[1]).toEqual({ role: Roles.Assistant, content: "chosen answer" });
  expect(chats.get(CHAT_A)?.messages[1]?.currentVersionIndex).toBe(1);
  expect(useChatStore.getState().messages[1]?.currentVersionIndex).toBe(1);
});
