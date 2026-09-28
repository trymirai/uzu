import type { ChatService } from ".";
import { emptyStats } from "./empty-stats";
import { noopUnsubscribe } from "../shared/noop";
import type { LlmAsyncStream } from "@/types/llm-stream";

export const webChat: ChatService = {
  runStream(): LlmAsyncStream {
    const stream = new ReadableStream<string>({
      start(c) {
        c.close();
      },
    });
    const result = Promise.resolve({
      text: "",
      error: "Chat not available on web",
      stats: emptyStats(),
    });
    return { runId: "", stream, result, cancel: () => Promise.resolve(), onParsed: noopUnsubscribe };
  },
  cancelRun: () => Promise.resolve(),
  generateTitle: () => Promise.reject(new Error("Not available on web")),
  cancelTitleGen: () => Promise.resolve(),
  getSamplingDefaults: () => Promise.resolve(null),
};
