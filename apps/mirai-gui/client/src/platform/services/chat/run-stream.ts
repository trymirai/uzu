import type {
  LlmAsyncStream,
  LlmRunParams,
  LlmRunResult,
  ParsedPatch,
  SessionOutputFinishReason,
  SessionOutputStats,
  TranscriptItem,
} from "@/types/llm-stream";
import { v4 as uuidv4 } from "uuid";
import { emptyStats } from "./empty-stats";

export type RunEvent =
  | { type: "chunk"; delta: string; parsed?: ParsedPatch }
  | { type: "chatName"; name: string }
  | { type: "transcript"; items: TranscriptItem[] }
  | { type: "transcriptDelta"; index: number; delta: string }
  | {
      type: "done";
      text: string;
      stats: SessionOutputStats;
      finishReason?: SessionOutputFinishReason;
      parsed?: ParsedPatch;
      chatName?: string;
      transcript?: TranscriptItem[];
    }
  | { type: "error"; error?: string };

export type RunTransport = {
  /** Resolves when the backend has finished the run; events arrive through `onEvent` meanwhile. */
  start: (runId: string, params: LlmRunParams, onEvent: (event: RunEvent) => void) => Promise<void>;
  cancel: (runId: string) => Promise<void>;
};

export function runLlmStream(transport: RunTransport, params: LlmRunParams): LlmAsyncStream {
  const runId = uuidv4();
  const parsedListeners = new Set<(patch: ParsedPatch) => void>();
  const chatNameListeners = new Set<(name: string) => void>();
  const transcriptListeners = new Set<(items: TranscriptItem[]) => void>();
  let chatName: string | undefined;
  let transcript: TranscriptItem[] | undefined;
  const pendingDeltas = new Map<number, string[]>();
  let transcriptFrame: number | undefined;
  let settled = false;
  let resolveResult!: (result: LlmRunResult) => void;
  const result = new Promise<LlmRunResult>((resolve) => {
    resolveResult = resolve;
  });

  // The first terminal outcome wins; anything the backend sends afterwards is dropped.
  const settle = (outcome: LlmRunResult): boolean => {
    if (settled) return false;
    flushTranscript();
    settled = true;
    resolveResult({
      ...outcome,
      ...(chatName !== undefined ? { chatName } : {}),
      ...(transcript !== undefined ? { transcript } : {}),
    });
    return true;
  };

  // Nothing else settles `result` once the reader is gone, so every awaiter would hang.
  const settleCanceled = () => settle({ text: "", finishReason: "Cancelled", stats: emptyStats() });

  const notifyParsed = (patch: ParsedPatch | undefined) => {
    if (patch) parsedListeners.forEach((listener) => listener(patch));
  };

  const notifyChatName = (name: string) => {
    if (name === chatName) return;
    chatName = name;
    chatNameListeners.forEach((listener) => listener(name));
  };

  const notifyTranscript = (items: TranscriptItem[]) => {
    if (transcriptFrame !== undefined) cancelAnimationFrame(transcriptFrame);
    transcriptFrame = undefined;
    pendingDeltas.clear();
    transcript = items;
    transcriptListeners.forEach((listener) => listener(items));
  };

  // IPC can deliver many tokens before the browser gets a chance to paint.
  // Keep that work proportional to the deltas; publish one immutable snapshot
  // per frame, and always publish the final tail before settling the run.
  const flushTranscript = () => {
    if (!transcript || pendingDeltas.size === 0) return;
    const items = [...transcript];
    for (const [index, deltas] of pendingDeltas) {
      const item = items[index];
      if (item?.type === "text" || item?.type === "thinking") {
        items[index] = { ...item, text: item.text + deltas.join("") };
      }
    }
    notifyTranscript(items);
  };

  const stream = new ReadableStream<string>({
    start(controller) {
      const fail = (message: string) => {
        if (!settle({ text: "", error: message, stats: emptyStats() })) return;
        controller.error(new Error(message));
      };

      const onEvent = (event: RunEvent) => {
        if (settled) return;
        switch (event.type) {
          case "chunk":
            controller.enqueue(event.delta ?? "");
            notifyParsed(event.parsed);
            return;
          case "chatName":
            notifyChatName(event.name);
            return;
          case "transcript":
            notifyTranscript(event.items);
            return;
          case "transcriptDelta": {
            const item = transcript?.[event.index];
            if (!transcript || !item || (item.type !== "text" && item.type !== "thinking")) {
              fail(`Invalid transcript delta at index ${event.index}`);
              return;
            }
            const deltas = pendingDeltas.get(event.index);
            if (deltas) deltas.push(event.delta);
            else pendingDeltas.set(event.index, [event.delta]);
            transcriptFrame ??= requestAnimationFrame(flushTranscript);
            return;
          }
          case "done":
            if (event.chatName !== undefined) notifyChatName(event.chatName);
            if (event.transcript !== undefined) notifyTranscript(event.transcript);
            settle({
              text: event.text,
              stats: event.stats,
              finishReason: event.finishReason,
              parsed: event.parsed,
            });
            controller.close();
            notifyParsed(event.parsed);
            return;
          case "error":
            fail(event.error || "Unknown error");
        }
      };

      // Run failures arrive as an error event; a rejection means the call itself
      // never reached the backend.
      transport.start(runId, params, onEvent).catch((e: unknown) => fail(String(e)));
    },
    cancel() {
      settleCanceled();
    },
  });

  return {
    runId,
    stream,
    result,
    cancel: async () => {
      try {
        await transport.cancel(runId);
      } finally {
        settleCanceled();
      }
    },
    onParsed: (listener) => {
      parsedListeners.add(listener);
      return () => parsedListeners.delete(listener);
    },
    onChatName: (listener) => {
      chatNameListeners.add(listener);
      if (chatName !== undefined) listener(chatName);
      return () => chatNameListeners.delete(listener);
    },
    onTranscript: (listener) => {
      flushTranscript();
      transcriptListeners.add(listener);
      if (transcript !== undefined) listener(transcript);
      return () => transcriptListeners.delete(listener);
    },
  };
}
