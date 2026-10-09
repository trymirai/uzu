import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { getPlatform } from "@/platform/platform-singleton";
import type { Message } from "@/types/message";
import { withParsedOutput } from "@/stores/chat/message-patches";
import { getRunMetrics } from "../services/run-metrics";
import type {
  LlmAsyncStream,
  LlmRunParams,
  OutputShape,
  SessionOutputFinishReason,
  SessionOutputStats,
} from "@/types/llm-stream";
import { useCallback, useRef, useState } from "react";
import { createRevealLoop } from "../services/reveal-loop";

export type StartStreamOptions = {
  repoId: string;
  messages: LlmRunParams["messages"];
  modelChatNamingEnabled?: boolean;
  dateTimeToolEnabled?: boolean;
  chartToolEnabled?: boolean;
  signal?: AbortSignal;
  messageId: string;
  chatId: string;
  updateText: (messageId: string, updatedText: string) => void;
  onError: (messageId: string, errorText: string) => void;
  onDone?: () => void;
  onFinishReason?: (reason: SessionOutputFinishReason) => void;
};

const activeVersionIndex = (message: Message | undefined): number | undefined =>
  message?.versions?.length ? (message.currentVersionIndex ?? message.versions.length - 1) : undefined;

const outputPatch = (message: Message, output: OutputShape, text?: string): Partial<Message> => {
  const versionIndex = activeVersionIndex(message);
  const patch = { output, ...(text !== undefined ? { text } : {}) };
  if (versionIndex === undefined) return patch;
  return { versions: message.versions?.map((v, i) => (i === versionIndex ? { ...v, ...patch } : v)) };
};

const applyOutput = (messageId: string, output: OutputShape): void => {
  const store = useChatStore.getState();
  const target = store.messages.find((m) => m.id === messageId);
  if (target) store.updateMessage(messageId, outputPatch(target, output));
};

const applyPerf = (messageId: string, stats: SessionOutputStats): void => {
  const store = useChatStore.getState();
  const versionIndex = activeVersionIndex(store.messages.find((m) => m.id === messageId));
  const metrics = getRunMetrics(stats);
  if (versionIndex === undefined) {
    store.updateMessagePerf(messageId, metrics, stats);
  } else {
    store.updateVersionPerf(messageId, versionIndex, metrics, stats);
  }
};

export const useLlmStream = (chatId: string) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const currentLlmRef = useRef<LlmAsyncStream | null>(null);
  const currentReaderRef = useRef<ReadableStreamDefaultReader<string> | null>(null);
  const currentRevealCancelRef = useRef<(() => void) | null>(null);

  const activeRunIdRef = useRef<string | null>(null);
  const canceledRunIdRef = useRef<string | null>(null);
  const finalize = useCallback((onDone?: () => void) => {
    setIsStreaming(false);
    setIsLoading(false);
    currentLlmRef.current = null;
    currentReaderRef.current = null;
    currentRevealCancelRef.current = null;
    useChatSessionStore.getState().setGenerating(false);
    useChatSessionStore.getState().clearActiveGenerating();
    onDone?.();
  }, []);

  const releaseCurrentRun = useCallback(async () => {
    currentRevealCancelRef.current?.();
    currentRevealCancelRef.current = null;
    const reader = currentReaderRef.current;
    const llm = currentLlmRef.current;
    if (reader) {
      await reader.cancel().catch(() => {});
      currentReaderRef.current = null;
    }
    if (llm) {
      await llm.cancel().catch(() => {});
      currentLlmRef.current = null;
    }
  }, []);

  const cancel = useCallback(async () => {
    const { canStop, withOperation } = useChatSessionStore.getState();
    if (!canStop()) return;
    await withOperation("stopping", async () => {
      canceledRunIdRef.current = activeRunIdRef.current;
      setIsStreaming(false);
      setIsLoading(false);
      if (currentLlmRef.current) {
        await releaseCurrentRun();
        return;
      }
      // A run outlives the page that started it: after a remount this instance
      // holds no handle, and the backend run is reachable only by its id.
      await useChatSessionStore.getState().cancelActiveRunForChat(chatId);
    });
  }, [chatId, releaseCurrentRun]);

  const startStream = useCallback(
    async (options: StartStreamOptions) => {
      const { messageId, chatId, repoId } = options;
      const session = useChatSessionStore.getState();

      if (session.operationState !== "running") throw new Error("startStream called without the run operation");

      const stopIfAborted = () => {
        if (!options.signal?.aborted) return false;
        options.onDone?.();
        return true;
      };
      if (stopIfAborted()) return;

      await releaseCurrentRun();
      if (stopIfAborted()) return;

      const modelChatNamingEnabled =
        options.modelChatNamingEnabled !== false && (await getPlatform().settings.getModelChatNamingEnabled());
      if (stopIfAborted()) return;
      // Compare against the title at the start of this run so a manual rename
      // made while the model thinks wins over its automatic naming.
      let expectedTitle = modelChatNamingEnabled
        ? (await getPlatform().storage.loadChat(chatId))?.metadata.title
        : undefined;
      if (stopIfAborted()) return;

      session.setGenerating(true);
      session.setActiveGenerating(chatId, messageId);

      if (options.messages.length === 0) {
        options.onError(messageId, "Error: empty message history");
        finalize(options.onDone);
        return;
      }

      const llm = useChatStore.getState().runChatStream({
        repoId,
        messages: options.messages,
        modelChatNamingEnabled,
        dateTimeToolEnabled: options.dateTimeToolEnabled,
        chartToolEnabled: options.chartToolEnabled,
      });
      const runId = llm.runId;
      const reader = llm.stream.getReader();
      currentLlmRef.current = llm;
      currentReaderRef.current = reader;
      activeRunIdRef.current = runId;
      session.setActiveRunId(runId);
      const isCurrentRun = () => activeRunIdRef.current === runId;

      const storeText = useChatStore.getState().messages.find((m) => m.id === messageId)?.text || "";
      const bufferText = useChatSessionStore.getState().activeAssistantMessageText || "";
      let transcriptText: string | undefined;
      const reveal = createRevealLoop({
        baseText: bufferText.length >= storeText.length ? bufferText : storeText,
        isActive: isCurrentRun,
        apply: (visibleText) => {
          options.updateText(messageId, visibleText);
          session.setActiveAssistantMessageText(visibleText);
        },
      });
      currentRevealCancelRef.current = reveal.cancel;

      let titleUpdates = Promise.resolve();
      const offChatName = llm.onChatName((name) => {
        if (!modelChatNamingEnabled) return;
        titleUpdates = titleUpdates.then(async () => {
          if (expectedTitle === undefined) return;
          try {
            await useChatStore.getState().updateChatTitle(chatId, name, expectedTitle);
            expectedTitle = name;
          } catch (error) {
            console.error("[storage] failed to save model chat name", { chatId }, error);
            useChatStore.setState((s) => ({ saveFailureCount: s.saveFailureCount + 1 }));
          }
        });
      });

      // Reasoning arrives as parsed patches before any text does.
      let hasContent = false;
      let output: OutputShape = {};
      const updateOutput = (next: OutputShape) => {
        output = next;
        applyOutput(messageId, output);
        session.setActiveAssistantMessageOutput(output);
      };
      const offParsed = llm.onParsed((patch) => {
        if (!hasContent && (patch.chainOfThought || patch.response)) {
          hasContent = true;
          setIsLoading(false);
        }
        updateOutput(withParsedOutput(output, patch));
      });
      const offTranscript = llm.onTranscript((items) => {
        if (items.length > 0) {
          hasContent = true;
          setIsLoading(false);
        }
        // The transcript is already visible. Keep the saved text and subsequent
        // model history at the same point, without waiting for the legacy pacer.
        reveal.cancel();
        transcriptText = items
          .flatMap((item) => (item.type === "text" && item.text.length > 0 ? [item.text] : []))
          .join("\n\n");
        const chainOfThought = items
          .flatMap((item) => (item.type === "thinking" && item.text.length > 0 ? [item.text] : []))
          .join("\n\n");
        options.updateText(messageId, transcriptText);
        session.setActiveAssistantMessageText(transcriptText);
        updateOutput({ ...withParsedOutput(output, { response: transcriptText, chainOfThought }), transcript: items });
      });

      setIsStreaming(true);
      setIsLoading(!hasContent);

      const fail = (message: string) => {
        // The pump may still be awaiting a read that resolves after this;
        // retiring the run id keeps it from finalizing a second time.
        activeRunIdRef.current = null;
        options.onError(messageId, `Error: ${message}`);
        if (transcriptText !== undefined) options.updateText(messageId, transcriptText);
        reveal.cancel();
        offParsed();
        offChatName();
        offTranscript();
        finalize(options.onDone);
      };

      let hasText = false;
      const pump = async (): Promise<void> => {
        for (;;) {
          const { value, done } = await reader.read();
          if (!isCurrentRun()) {
            reveal.cancel();
            offParsed();
            return;
          }
          if (done) {
            if (output.transcript === undefined) await reveal.drain();
            offParsed();
            if (!isCurrentRun()) return;
            session.setActiveAssistantMessageText(null);
            finalize(options.onDone);
            return;
          }
          if (value) {
            if (!hasText) setIsLoading(false);
            hasText = true;
            if (output.transcript === undefined) reveal.append(value);
          }
        }
      };

      try {
        await pump();
      } catch (e) {
        // A reader abort caused by Stop or by a newer run is not a failure.
        if (canceledRunIdRef.current === runId || !isCurrentRun()) {
          canceledRunIdRef.current = null;
          reveal.cancel();
          offParsed();
        } else {
          fail(e instanceof Error ? e.message : String(e));
        }
      }

      const result = await llm.result;
      offChatName();
      offTranscript();
      await titleUpdates;
      // Stop pressed after leaving and reopening the chat: the stop handler saved
      // the text at that moment, and this stream kept printing after it.
      const wasCanceled = canceledRunIdRef.current === runId || result.finishReason === "Cancelled";
      if (wasCanceled || result.error) {
        canceledRunIdRef.current = null;
        const current =
          useChatStore.getState().messages.find((m) => m.id === messageId) ??
          (await getPlatform().storage.loadChat(chatId))?.messages?.find((m) => m.id === messageId);
        if (current) {
          const patch = {
            text: transcriptText ?? current.text,
            versions: current.versions,
            error: current.error,
            output: current.output,
            ...(Object.keys(output).length > 0 ? outputPatch(current, output, transcriptText) : {}),
          };
          await useChatStore.getState().persistMessagePatch(chatId, messageId, patch);
        }
        return result;
      }
      if (!isCurrentRun()) return;

      applyPerf(messageId, result.stats);
      await useChatStore
        .getState()
        .finalizeAssistantMessage(chatId, messageId, result.text || "", result.parsed, result.transcript);
      if (result.finishReason) options.onFinishReason?.(result.finishReason);
      return result;
    },
    [finalize, releaseCurrentRun],
  );

  return { isStreaming, isLoading, cancel, startStream };
};
