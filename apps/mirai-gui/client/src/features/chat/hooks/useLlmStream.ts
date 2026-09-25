import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import type { Message } from "@/types/message";
import {
  getRunMetrics,
  withParsedOutput,
  type LlmAsyncStream,
  type LlmRunParams,
  type ParsedPatch,
  type SessionOutputFinishReason,
  type SessionOutputStats,
} from "@/utils/llm-stream";
import { useCallback, useRef, useState } from "react";
import { createRevealLoop } from "../services/revealLoop";

export type StartStreamOptions = {
  repoId: string;
  messages: LlmRunParams["messages"];
  messageId: string;
  chatId: string;
  updateText: (messageId: string, updatedText: string) => void;
  onError: (messageId: string, errorText: string) => void;
  onDone?: () => void;
  onFinishReason?: (reason: SessionOutputFinishReason) => void;
};

export const STALL_TIMEOUT_MS = 15000;

const activeVersionIndex = (message: Message | undefined): number | undefined =>
  message?.versions?.length ? (message.currentVersionIndex ?? message.versions.length - 1) : undefined;

const applyParsed = (messageId: string, patch: ParsedPatch): void => {
  const store = useChatStore.getState();
  const target = store.messages.find((m) => m.id === messageId);
  const versionIndex = activeVersionIndex(target);
  if (versionIndex === undefined) {
    store.updateMessage(messageId, { output: withParsedOutput(target?.output, patch) });
    return;
  }
  const versions = (target?.versions || []).map((v, i) =>
    i === versionIndex ? { ...v, output: withParsedOutput(v.output, patch) } : v,
  );
  store.updateMessage(messageId, { versions });
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
  const stallTimeoutRef = useRef<number | null>(null);

  const clearStallTimeout = useCallback(() => {
    if (stallTimeoutRef.current === null) return;
    window.clearTimeout(stallTimeoutRef.current);
    stallTimeoutRef.current = null;
  }, []);

  const finalize = useCallback(
    (onDone?: () => void) => {
      clearStallTimeout();
      setIsStreaming(false);
      setIsLoading(false);
      currentLlmRef.current = null;
      currentReaderRef.current = null;
      currentRevealCancelRef.current = null;
      useChatSessionStore.getState().setGenerating(false);
      useChatSessionStore.getState().clearActiveGenerating();
      onDone?.();
    },
    [clearStallTimeout],
  );

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
      const residentSession = useRuntimeSessionStore.getState().residentSession;

      if (!session.canRun({ repoId })) {
        options.onError(messageId, "Error: Model is busy");
        return;
      }

      return session.withOperation("running", async () => {
        await releaseCurrentRun();

        if (residentSession && residentSession.repoId !== repoId) {
          await useChatStore.getState().ejectChatSession(residentSession.repoId);
        }

        session.setGenerating(true);
        session.setActiveGenerating(chatId, messageId);

        if (options.messages.length === 0) {
          options.onError(messageId, "Error: empty message history");
          finalize(options.onDone);
          return;
        }

        const llm = useChatStore.getState().runChatStream({ repoId, messages: options.messages });
        const runId = llm.runId;
        const reader = llm.stream.getReader();
        currentLlmRef.current = llm;
        currentReaderRef.current = reader;
        activeRunIdRef.current = runId;
        session.setActiveRunId(runId);
        const isCurrentRun = () => activeRunIdRef.current === runId;

        // Reasoning arrives as parsed patches before any text does.
        let hasContent = false;
        const offParsed = llm.onParsed((patch) => {
          if (!hasContent && (patch.chainOfThought || patch.response)) {
            hasContent = true;
            setIsLoading(false);
          }
          applyParsed(messageId, patch);
        });

        setIsStreaming(true);
        setIsLoading(true);

        const storeText = useChatStore.getState().messages.find((m) => m.id === messageId)?.text || "";
        const bufferText = useChatSessionStore.getState().activeAssistantMessageText || "";
        const reveal = createRevealLoop({
          baseText: bufferText.length >= storeText.length ? bufferText : storeText,
          isActive: isCurrentRun,
          apply: (visibleText) => {
            options.updateText(messageId, visibleText);
            session.setActiveAssistantMessageText(visibleText);
          },
        });
        currentRevealCancelRef.current = reveal.cancel;

        const fail = (message: string) => {
          // The pump may still be awaiting a read that resolves after this;
          // retiring the run id keeps it from finalizing a second time.
          activeRunIdRef.current = null;
          options.onError(messageId, `Error: ${message}`);
          reveal.cancel();
          offParsed();
          finalize(options.onDone);
        };

        let hasText = false;
        const armStallTimeout = () => {
          if (!hasText) return;
          clearStallTimeout();
          stallTimeoutRef.current = window.setTimeout(() => {
            if (!isCurrentRun()) return;
            canceledRunIdRef.current = runId;
            void reader.cancel();
            void llm.cancel();
            fail("Stream timeout");
          }, STALL_TIMEOUT_MS);
        };

        const pump = async (): Promise<void> => {
          for (;;) {
            const { value, done } = await reader.read();
            if (!isCurrentRun()) {
              reveal.cancel();
              offParsed();
              return;
            }
            if (done) {
              clearStallTimeout();
              await reveal.drain();
              offParsed();
              if (!isCurrentRun()) return;
              session.setActiveAssistantMessageText(null);
              finalize(options.onDone);
              return;
            }
            if (value) {
              if (!hasText) setIsLoading(false);
              hasText = true;
              reveal.append(value);
            }
            armStallTimeout();
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
        if (result.error) return;
        // A canceled run was already persisted by the stop handler.
        const wasCanceled = canceledRunIdRef.current === runId;
        if (wasCanceled) canceledRunIdRef.current = null;
        if (wasCanceled || !isCurrentRun()) return;

        applyPerf(messageId, result.stats);
        await useChatStore.getState().finalizeAssistantMessage(chatId, messageId, result.text || "", result.parsed);
        if (result.finishReason) options.onFinishReason?.(result.finishReason);
      });
    },
    [clearStallTimeout, finalize, releaseCurrentRun],
  );

  return { isStreaming, isLoading, cancel, startStream };
};
