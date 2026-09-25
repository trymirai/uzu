import type {
  LlmAsyncStream,
  LlmRunParams,
  LlmRunResult,
  ParsedPatch,
  SessionOutputFinishReason,
  SessionOutputStats,
} from "@/utils/llm-stream";
import { v4 as uuidv4 } from "uuid";
import { emptyStats } from "./emptyStats";

export type RunEvent =
  | { type: "chunk"; delta: string; parsed?: ParsedPatch }
  | {
      type: "done";
      text: string;
      stats: SessionOutputStats;
      finishReason?: SessionOutputFinishReason;
      parsed?: ParsedPatch;
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
  let settled = false;
  let resolveResult!: (result: LlmRunResult) => void;
  const result = new Promise<LlmRunResult>((resolve) => {
    resolveResult = resolve;
  });

  // The first terminal outcome wins; anything the backend sends afterwards is dropped.
  const settle = (outcome: LlmRunResult): boolean => {
    if (settled) return false;
    settled = true;
    resolveResult(outcome);
    return true;
  };

  // Nothing else settles `result` once the reader is gone, so every awaiter would hang.
  const settleCanceled = () => settle({ text: "", finishReason: "Cancelled", stats: emptyStats() });

  const notifyParsed = (patch: ParsedPatch | undefined) => {
    if (patch) parsedListeners.forEach((listener) => listener(patch));
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
          case "done":
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
  };
}
