import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { runLlmStream, type RunEvent, type RunTransport } from "@/platform/services/chat/run-stream";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { Roles } from "@/types/chat";
import type { SessionOutputStats } from "@/types/llm-stream";
import { STALL_TIMEOUT_MS, useLlmStream, type StartStreamOptions } from "./use-llm-stream";

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
  const transport: RunTransport = {
    start: (_runId, _params, onEvent) => {
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
  return { hook, options, callbacks, transport, finalizeAssistantMessage, emit: (event: RunEvent) => emit(event) };
};

const messageText = () => useChatStore.getState().messages.find((m) => m.id === MESSAGE_ID)?.text;

describe("useLlmStream", () => {
  beforeEach(() => {
    useChatSessionStore.setState(sessionDefaults, true);
    useChatStore.setState(chatDefaults, true);
  });

  it("reveals streamed text, records perf and finalizes the message", async () => {
    const { hook, options, callbacks, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = hook.result.current.startStream(options);
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
    expect(finalizeAssistantMessage).toHaveBeenCalledWith(CHAT_ID, MESSAGE_ID, "Hello world", {
      response: "Hello world",
    });
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

  it("reports a backend error and does not finalize", async () => {
    const { hook, options, callbacks, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = hook.result.current.startStream(options);
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
    const { hook, options, callbacks, transport, finalizeAssistantMessage, emit } = setup();

    let run!: Promise<unknown>;
    act(() => {
      run = hook.result.current.startStream(options);
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
    const { hook, options, transport } = setup();
    const cancelActiveRunForChat = vi.fn(() => Promise.resolve());
    useChatSessionStore.setState({ cancelActiveRunForChat });

    act(() => {
      void hook.result.current.startStream(options);
    });
    await waitFor(() => expect(hook.result.current.isStreaming).toBe(true));

    const remounted = renderHook(() => useLlmStream(CHAT_ID));
    await act(() => remounted.result.current.cancel());

    expect(cancelActiveRunForChat).toHaveBeenCalledWith(CHAT_ID);
    expect(transport.cancel).not.toHaveBeenCalled();
  });

  it("reports a stalled stream once", async () => {
    vi.useFakeTimers();
    try {
      const { hook, options, callbacks, emit } = setup();

      act(() => {
        void hook.result.current.startStream(options);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      act(() => emit({ type: "chunk", delta: "partial" }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(STALL_TIMEOUT_MS + 1);
      });

      expect(callbacks.onError).toHaveBeenCalledWith(MESSAGE_ID, "Error: Stream timeout");
      expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refuses an empty history", async () => {
    const { hook, options, callbacks, finalizeAssistantMessage } = setup();

    await act(() => hook.result.current.startStream({ ...options, messages: [] }));

    expect(callbacks.onError).toHaveBeenCalledWith(MESSAGE_ID, "Error: empty message history");
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    expect(finalizeAssistantMessage).not.toHaveBeenCalled();
    expect(useChatSessionStore.getState().isGenerating).toBe(false);
  });

  it("refuses to start while another generation is running", async () => {
    const { hook, options, callbacks } = setup();
    useChatSessionStore.setState({ isGenerating: true });

    await act(() => hook.result.current.startStream(options));

    expect(callbacks.onError).toHaveBeenCalledWith(MESSAGE_ID, "Error: Model is busy");
    expect(hook.result.current.isStreaming).toBe(false);
  });
});
