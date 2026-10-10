import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { runLlmStream, type RunEvent, type RunTransport } from "@/platform/services/chat/run-stream";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import type { ChatMetadata } from "@/platform/services/storage";
import { Roles } from "@/types/chat";
import type { SessionOutputStats, TranscriptItem } from "@/types/llm-stream";
import type { Message } from "@/types/message";
import { useLlmStream, type StartStreamOptions } from "./use-llm-stream";

const mocks = vi.hoisted(() => ({
  getModelChatNamingEnabled: vi.fn(async () => true),
  cancelRun: vi.fn(async () => {}),
  updateStoredMessage: vi.fn(async () => {}),
  updateChatTitle: vi.fn<(id: string, name: string, expected?: string) => Promise<boolean>>(async () => true),
  listChats: vi.fn<() => Promise<ChatMetadata[]>>(async () => []),
  loadChat: vi.fn<() => Promise<{ metadata: { title: string }; messages?: Message[] }>>(async () => ({
    metadata: { title: "Untitled" },
  })),
}));
vi.mock("@/platform/platform-singleton", () => ({
  getPlatform: () => ({
    chat: { cancelRun: mocks.cancelRun },
    settings: { getModelChatNamingEnabled: mocks.getModelChatNamingEnabled },
    storage: {
      updateStoredMessage: mocks.updateStoredMessage,
      loadChat: mocks.loadChat,
      updateChatTitle: mocks.updateChatTitle,
      listChats: mocks.listChats,
    },
  }),
}));

const CHAT_ID = "chat-1";
const MESSAGE_ID = "assistant-1";

const stats: SessionOutputStats = {
  prefillStats: { duration: 0.25, tokensCount: 5, tokensPerSecond: 20 },
  generateStats: { duration: 2, tokensCount: 40, tokensPerSecond: 20 },
  totalStats: { duration: 2.25, tokensCountInput: 5, tokensCountOutput: 40 },
};

const sessionDefaults = useChatSessionStore.getState();
const chatDefaults = useChatStore.getState();

const setup = () => {
  let emit: (event: RunEvent) => void = () => {
    throw new Error("run was not started");
  };
  let startedRunId = "";
  const transport: RunTransport = {
    start: (runId, _params, onEvent) => {
      startedRunId = runId;
      emit = onEvent;
      return new Promise(() => {});
    },
    cancel: vi.fn(() => Promise.resolve()),
  };
  const finalizeAssistantMessage = vi.fn(() => Promise.resolve());
  useChatStore.setState({
    currentChatId: CHAT_ID,
    messages: [{ id: MESSAGE_ID, text: "", sender: Roles.Assistant, timestamp: 1 }],
    runChatStream: (params) => runLlmStream(transport, params),
    finalizeAssistantMessage,
  });

  const callbacks = {
    updateText: vi.fn((id: string, text: string) => useChatStore.getState().updateMessage(id, { text })),
    onError: vi.fn(),
    onDone: vi.fn(),
    onFinishReason: vi.fn(),
  };
  const options: StartStreamOptions = {
    repoId: "vendor/model",
    chatId: CHAT_ID,
    messageId: MESSAGE_ID,
    messages: [{ role: "user", content: "hi" }],
    ...callbacks,
  };
  const hook = renderHook(() => useLlmStream(CHAT_ID));
  const start = (o: StartStreamOptions) =>
    useChatSessionStore.getState().withOperation("running", () => hook.result.current.startStream(o));
  return {
    hook,
    start,
    options,
    callbacks,
    transport,
    finalizeAssistantMessage,
    emit: (event: RunEvent) => emit(event),
    runId: () => startedRunId,
  };
};

const messageText = () => useChatStore.getState().messages.find((m) => m.id === MESSAGE_ID)?.text;

describe("useLlmStream", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getModelChatNamingEnabled.mockResolvedValue(true);
    mocks.listChats.mockResolvedValue([]);
    mocks.loadChat.mockResolvedValue({ metadata: { title: "Untitled" } });
    useChatSessionStore.setState(sessionDefaults, true);
    useChatStore.setState(chatDefaults, true);
  });

  it("enforces the global naming switch even with an explicit stream opt-in", async () => {
    mocks.getModelChatNamingEnabled.mockResolvedValue(false);
    const { hook, start, options, emit } = setup();
    const runChatStream = vi.spyOn(useChatStore.getState(), "runChatStream");
    let run!: Promise<unknown>;
    act(() => {
      run = start({ ...options, modelChatNamingEnabled: true });
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    expect(runChatStream).toHaveBeenCalledWith(expect.objectContaining({ modelChatNamingEnabled: false }));
    expect(mocks.loadChat).not.toHaveBeenCalled();
    act(() => {
      emit({ type: "chatName", name: "Unexpected name" });
      emit({ type: "done", text: "Answer", stats, finishReason: "Stop" });
    });
    await act(() => run);
    expect(mocks.updateChatTitle).not.toHaveBeenCalled();
  });

  it("does not start a native run when its chat operation is canceled while loading the saved title", async () => {
    const { hook, options, callbacks, runId } = setup();
    const runChatStream = vi.spyOn(useChatStore.getState(), "runChatStream");
    let resolveTitle!: (value: { metadata: { title: string } }) => void;
    mocks.loadChat.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveTitle = resolve;
      }),
    );
    let run!: Promise<unknown>;
    act(() => {
      run = useChatSessionStore
        .getState()
        .withOperation("running", (signal) => hook.result.current.startStream({ ...options, signal }), CHAT_ID);
    });
    await waitFor(() => expect(mocks.loadChat).toHaveBeenCalled());
    expect(useChatSessionStore.getState().activeRunId).toBeNull();
    await act(() => useChatSessionStore.getState().cancelActiveRunForChat(CHAT_ID));
    await act(async () => {
      resolveTitle({ metadata: { title: "Untitled" } });
      await run;
    });

    expect(runChatStream).not.toHaveBeenCalled();
    expect(runId()).toBe("");
    expect(callbacks.onDone).toHaveBeenCalledOnce();
    expect(callbacks.onError).not.toHaveBeenCalled();
    expect(hook.result.current.isStreaming).toBe(false);
    expect(useChatSessionStore.getState()).toMatchObject({
      isGenerating: false,
      activeRunId: null,
      operationState: "idle",
    });
  });

  it("reveals streamed text, records perf and finalizes the message", async () => {
    const { hook, start, options, callbacks, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    expect(hook.result.current.isLoading).toBe(true);
    expect(useChatSessionStore.getState().isGenerating).toBe(true);

    act(() => emit({ type: "chunk", delta: "", parsed: { chainOfThought: "let me think" } }));
    await waitFor(() => expect(hook.result.current.isLoading).toBe(false));
    act(() => {
      emit({ type: "chunk", delta: "Hello" });
      emit({ type: "chunk", delta: " world" });
      emit({ type: "done", text: "Hello world", stats, finishReason: "Stop", parsed: { response: "Hello world" } });
    });
    await act(() => run);

    expect(messageText()).toBe("Hello world");
    const message = useChatStore.getState().messages.find((m) => m.id === MESSAGE_ID);
    expect(message?.output?.text?.parsed).toEqual({ chainOfThought: "let me think", response: "Hello world" });
    expect(message?.perf).toEqual({ ttftSec: 0.25, totalSec: 2.25, tps: 20, tokensOut: 40 });
    expect(finalizeAssistantMessage).toHaveBeenCalledWith(
      CHAT_ID,
      MESSAGE_ID,
      "Hello world",
      {
        response: "Hello world",
      },
      undefined,
    );
    expect(callbacks.onFinishReason).toHaveBeenCalledWith("Stop");
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    expect(callbacks.onError).not.toHaveBeenCalled();
    expect(hook.result.current.isStreaming).toBe(false);
    expect(useChatSessionStore.getState()).toMatchObject({
      isGenerating: false,
      activeGeneratingChatId: null,
      activeAssistantMessageText: null,
      operationState: "idle",
    });
  });

  it("keeps a tool-only partial reply when canceled and forwards tool preferences", async () => {
    const { hook, start, options, emit } = setup();
    const runChatStream = vi.spyOn(useChatStore.getState(), "runChatStream");
    let run!: Promise<unknown>;
    act(() => {
      run = start({ ...options, dateTimeToolEnabled: false, chartToolEnabled: false });
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    const transcript: TranscriptItem[] = [{ type: "toolCall", name: "get_current_date_time", called: true }];
    act(() => emit({ type: "transcript", items: transcript }));
    expect(hook.result.current.isLoading).toBe(false);
    expect(useChatStore.getState().messages[0]?.output?.transcript).toEqual(transcript);
    expect(useChatSessionStore.getState().activeAssistantMessageOutput?.transcript).toEqual(transcript);
    await act(() => hook.result.current.cancel());
    await act(() => run);
    expect(runChatStream).toHaveBeenCalledWith(
      expect.objectContaining({ dateTimeToolEnabled: false, chartToolEnabled: false }),
    );
    expect(mocks.updateStoredMessage).toHaveBeenCalledWith(
      CHAT_ID,
      MESSAGE_ID,
      expect.objectContaining({ output: expect.objectContaining({ transcript }) }),
    );
  });

  it.each(["cancel", "error"] as const)(
    "saves incrementally streamed text and reasoning on %s without legacy chunks",
    async (end) => {
      const { hook, start, options, emit } = setup();
      let run!: Promise<unknown>;
      act(() => {
        run = start({
          ...options,
          onError: (id, error) => useChatStore.getState().updateMessage(id, { text: "", error }),
        });
      });
      await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
      const transcript: TranscriptItem[] = [
        { type: "thinking", text: "Check the clock 🕰️", completed: true },
        { type: "text", text: "Before the tool." },
        { type: "toolCall", name: "get_current_date_time", called: true },
        { type: "text", text: "A long response. ".repeat(100) },
      ];
      const fullText = transcript.flatMap((item) => (item.type === "text" ? [item.text] : [])).join("\n\n");
      act(() => {
        emit({ type: "transcript", items: [{ type: "thinking", text: "Check" }] });
        emit({ type: "transcriptDelta", index: 0, delta: " the clock 🕰️" });
        emit({ type: "transcript", items: [...transcript.slice(0, -1), { type: "text", text: "" }] });
        emit({ type: "transcriptDelta", index: 3, delta: "A long response. ".repeat(100) });
      });
      // The last delta is still waiting for a paint; either terminal path must
      // flush it before saving, even if that frame never runs.
      expect(messageText()).toBe("Before the tool.");
      if (end === "cancel") await act(() => hook.result.current.cancel());
      else act(() => emit({ type: "error", error: "Model failed" }));
      await act(() => run);
      expect(messageText()).toBe(fullText);
      expect(mocks.updateStoredMessage).toHaveBeenCalledWith(
        CHAT_ID,
        MESSAGE_ID,
        expect.objectContaining({
          text: fullText,
          output: expect.objectContaining({
            transcript,
            text: { parsed: { response: fullText, chainOfThought: "Check the clock 🕰️" } },
          }),
        }),
      );
    },
  );

  it("clears superseded parsed reasoning when a snapshot promotes it to visible text", async () => {
    const { hook, start, options, emit } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => {
      emit({ type: "transcript", items: [{ type: "thinking", text: "Visible" }] });
      emit({ type: "transcriptDelta", index: 0, delta: " answer" });
    });
    await waitFor(() =>
      expect(useChatStore.getState().messages[0]?.output?.text?.parsed).toEqual({
        response: "",
        chainOfThought: "Visible answer",
      }),
    );
    act(() => emit({ type: "transcript", items: [{ type: "text", text: "Visible answer" }] }));
    expect(useChatStore.getState().messages[0]?.output?.text?.parsed).toEqual({
      response: "Visible answer",
      chainOfThought: "",
    });
    await act(() => hook.result.current.cancel());
    await act(() => run);
  });

  it("finalizes the complete transcript even after leaving the chat", async () => {
    const { hook, start, options, emit, finalizeAssistantMessage } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => useChatStore.setState({ currentChatId: "another-chat", messages: [] }));
    const transcript: TranscriptItem[] = [
      { type: "text", text: "Before" },
      { type: "toolCall", name: "get_current_date_time", called: true },
      { type: "text", text: "After" },
    ];
    act(() => emit({ type: "done", text: "Before\n\nAfter", stats, transcript }));
    await act(() => run);
    expect(finalizeAssistantMessage).toHaveBeenCalledWith(
      CHAT_ID,
      MESSAGE_ID,
      "Before\n\nAfter",
      undefined,
      transcript,
    );
    expect(useChatStore.getState().messages).toEqual([]);
  });

  it("saves the active version's visible text and transcript when stopped from another chat", async () => {
    const { hook, start, options, emit } = setup();
    const message: Message = {
      id: MESSAGE_ID,
      text: "",
      sender: "assistant",
      timestamp: 1,
      versions: [
        { id: "old", text: "Previous response", modelId: "model", modelName: "Model", timestamp: 1 },
        { id: "new", text: "", modelId: "model", modelName: "Model", timestamp: 2 },
      ],
      currentVersionIndex: 1,
    };
    useChatStore.setState({ messages: [message] });
    mocks.loadChat.mockResolvedValue({ metadata: { title: "Untitled" }, messages: [message] });
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => useChatStore.setState({ currentChatId: "other", messages: [] }));
    const transcript: TranscriptItem[] = [
      { type: "toolCall", name: "get_current_date_time", called: true },
      { type: "text", text: "New visible response" },
    ];
    act(() => {
      emit({ type: "transcript", items: [transcript[0]!, { type: "text", text: "New visible" }] });
      emit({ type: "transcriptDelta", index: 1, delta: " response" });
    });
    await act(() => hook.result.current.cancel());
    await act(() => run);
    expect(mocks.updateStoredMessage).toHaveBeenCalledWith(
      CHAT_ID,
      MESSAGE_ID,
      expect.objectContaining({
        text: "New visible response",
        versions: [
          message.versions![0],
          expect.objectContaining({ text: "New visible response", output: expect.objectContaining({ transcript }) }),
        ],
      }),
    );
  });

  it("reports a backend error and does not finalize", async () => {
    const { hook, start, options, callbacks, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => emit({ type: "error", error: "model exploded" }));
    await act(() => run);

    expect(callbacks.onError).toHaveBeenCalledWith(MESSAGE_ID, "Error: model exploded");
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    expect(finalizeAssistantMessage).not.toHaveBeenCalled();
    expect(hook.result.current.isStreaming).toBe(false);
    expect(useChatSessionStore.getState().isGenerating).toBe(false);
  });

  it("stops a run without finalizing or reporting an error", async () => {
    const { hook, start, options, callbacks, transport, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => emit({ type: "chunk", delta: "partial" }));
    await act(() => hook.result.current.cancel());
    await act(() => run);

    expect(transport.cancel).toHaveBeenCalledTimes(1);
    expect(finalizeAssistantMessage).not.toHaveBeenCalled();
    expect(callbacks.onError).not.toHaveBeenCalled();
    expect(callbacks.onFinishReason).not.toHaveBeenCalled();
    expect(hook.result.current.isStreaming).toBe(false);
    expect(useChatSessionStore.getState()).toMatchObject({ isGenerating: false, operationState: "idle" });
  });

  it("stops a run started by an instance that has since unmounted", async () => {
    const { hook, start, options, transport, finalizeAssistantMessage, emit, runId } = setup();
    mocks.cancelRun.mockClear();

    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => emit({ type: "chunk", delta: "partial" }));
    hook.unmount();

    const remounted = renderHook(() => useLlmStream(CHAT_ID));
    await act(() => remounted.result.current.cancel());
    expect(mocks.cancelRun).toHaveBeenCalledWith(runId());
    expect(transport.cancel).not.toHaveBeenCalled();

    act(() => emit({ type: "done", text: "partial", stats, finishReason: "Cancelled" }));
    await act(() => run);

    expect(finalizeAssistantMessage).not.toHaveBeenCalled();
    expect(mocks.updateStoredMessage).toHaveBeenCalledWith(
      CHAT_ID,
      MESSAGE_ID,
      expect.objectContaining({ text: "partial" }),
    );
    expect(useChatSessionStore.getState()).toMatchObject({ isGenerating: false, operationState: "idle" });
  });

  it("stops a run whose model is still loading", async () => {
    const { hook, start, options, transport } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => useChatSessionStore.getState().startModelLoading());

    await act(() => hook.result.current.cancel());
    await act(() => run);

    expect(transport.cancel).toHaveBeenCalledTimes(1);
    expect(hook.result.current.isStreaming).toBe(false);
  });

  it("keeps a quiet generation running until the backend finishes", async () => {
    vi.useFakeTimers();
    try {
      const { hook, start, options, callbacks, emit, transport } = setup();
      let run!: Promise<unknown>;
      act(() => {
        run = start(options);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      act(() => emit({ type: "chunk", delta: "partial" }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(120_000);
      });

      expect(hook.result.current.isStreaming).toBe(true);
      expect(callbacks.onError).not.toHaveBeenCalled();
      expect(callbacks.onDone).not.toHaveBeenCalled();
      expect(transport.cancel).not.toHaveBeenCalled();

      act(() => emit({ type: "done", text: "partial", stats, finishReason: "Stop" }));
      await act(() => run);
      expect(callbacks.onDone).toHaveBeenCalledTimes(1);
      expect(hook.result.current.isStreaming).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refuses an empty history", async () => {
    const { start, options, callbacks, finalizeAssistantMessage } = setup();

    await act(() => start({ ...options, messages: [] }));

    expect(callbacks.onError).toHaveBeenCalledWith(MESSAGE_ID, "Error: empty message history");
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    expect(finalizeAssistantMessage).not.toHaveBeenCalled();
    expect(useChatSessionStore.getState().isGenerating).toBe(false);
  });

  it("saves model names to the originating chat even after navigation and a later stream error", async () => {
    const { hook, start, options, emit } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => useChatStore.getState().createNewChat("another-chat"));

    act(() => emit({ type: "chatName", name: "First name" }));
    await waitFor(() => expect(mocks.updateChatTitle).toHaveBeenCalledWith(CHAT_ID, "First name", "Untitled"));
    act(() => {
      emit({ type: "chatName", name: "Changed topic" });
      emit({ type: "error", error: "later generation failure" });
    });
    await act(() => run);

    expect(mocks.updateChatTitle).toHaveBeenLastCalledWith(CHAT_ID, "Changed topic", "First name");
    expect(useChatStore.getState().currentChatId).toBe("another-chat");
  });

  it("keeps a manual rename made during a run", async () => {
    let storedName = "Untitled";
    mocks.updateChatTitle.mockImplementationOnce(async (_id, name, expected) => {
      if (storedName !== expected) return false;
      storedName = name;
      return true;
    });
    const { hook, start, options, emit } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    storedName = "My chosen name";

    act(() => {
      emit({ type: "chatName", name: "Automatic name" });
      emit({ type: "done", text: "Answer", stats, finishReason: "Stop" });
    });
    await act(() => run);

    expect(storedName).toBe("My chosen name");
  });

  it("stops automatic renaming after a rejected update even if its candidate matches a manual rename", async () => {
    const { hook, start, options, emit } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    mocks.updateChatTitle.mockResolvedValueOnce(false);

    act(() => {
      emit({ type: "chatName", name: "My manual title" });
      emit({ type: "chatName", name: "Another automatic title" });
      emit({ type: "done", text: "Answer", stats, chatName: "Another automatic title" });
    });
    await act(() => run);

    expect(mocks.updateChatTitle).toHaveBeenCalledExactlyOnceWith(CHAT_ID, "My manual title", "Untitled");
  });

  it("reports a failed name save without losing the response", async () => {
    mocks.updateChatTitle.mockRejectedValueOnce(new Error("disk full"));
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const { hook, start, options, callbacks, finalizeAssistantMessage, emit } = setup();
      let run!: Promise<unknown>;
      act(() => {
        run = start(options);
      });
      await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
      act(() => {
        emit({ type: "chatName", name: "Automatic name" });
        emit({ type: "done", text: "Answer", stats, finishReason: "Stop" });
      });
      await act(() => run);

      expect(useChatStore.getState().saveFailureCount).toBe(1);
      expect(callbacks.onError).not.toHaveBeenCalled();
      expect(finalizeAssistantMessage).toHaveBeenCalledWith(CHAT_ID, MESSAGE_ID, "Answer", undefined, undefined);
    } finally {
      logged.mockRestore();
    }
  });

  it("keeps the saved title visible and accepts the next correction after the chat list refresh fails", async () => {
    const { hook, start, options, emit } = setup();
    const metadata: ChatMetadata = {
      id: CHAT_ID,
      title: "Untitled",
      createdAt: 1,
      updatedAt: 1,
      messageCount: 2,
    };
    const other = { ...metadata, id: "other-chat", title: "Other chat" };
    useChatStore.setState({ savedChats: [metadata, other] });
    let storedTitle = "Untitled";
    const saveTitle = async (_id: string, next: string, expected?: string) => {
      if (expected !== storedTitle) return false;
      storedTitle = next;
      return true;
    };
    mocks.updateChatTitle.mockImplementationOnce(saveTitle).mockImplementationOnce(saveTitle);
    const first = "First accepted title";
    const corrected = "Corrected title";
    const refreshed = [other, { ...metadata, title: corrected, updatedAt: 2 }];
    mocks.listChats.mockRejectedValueOnce(new Error("read failed")).mockResolvedValueOnce(refreshed);
    const warned = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      let run!: Promise<unknown>;
      act(() => {
        run = start(options);
      });
      await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
      act(() => emit({ type: "chatName", name: first }));
      await waitFor(() => expect(warned).toHaveBeenCalledOnce());

      expect(useChatStore.getState().savedChats).toEqual([{ ...metadata, title: first }, other]);
      expect(useChatStore.getState().saveFailureCount).toBe(0);

      act(() => {
        emit({ type: "chatName", name: corrected });
        emit({ type: "done", text: "Answer", stats, chatName: corrected });
      });
      await act(() => run);

      expect(mocks.updateChatTitle).toHaveBeenLastCalledWith(CHAT_ID, corrected, first);
      expect(storedTitle).toEqual(corrected);
      expect(useChatStore.getState().savedChats).toEqual(refreshed);
      expect(useChatStore.getState().saveFailureCount).toBe(0);
    } finally {
      warned.mockRestore();
    }
  });

  it("keeps a name already set when the response is canceled", async () => {
    const { hook, start, options, emit } = setup();
    let run!: Promise<unknown>;
    act(() => {
      run = start(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));
    act(() => emit({ type: "chatName", name: "Accepted name" }));
    await act(() => hook.result.current.cancel());
    await act(() => run);

    expect(mocks.updateChatTitle).toHaveBeenCalledWith(CHAT_ID, "Accepted name", "Untitled");
  });
});
