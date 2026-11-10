import type { ChatService } from "@/platform/services/chat";
import { runLlmStream } from "@/platform/services/chat/run-stream";
import type { SessionService } from "@/platform/services/session";
import { BrowserChatRuntime } from "./browser-runtime";

export const browserChatRuntime = new BrowserChatRuntime();

export const browserChat: ChatService = {
  runStream: (params) => runLlmStream(browserChatRuntime.transport, params),
  cancelRun: (runId) => browserChatRuntime.transport.cancel(runId),
  generateTitle: (params) => browserChatRuntime.generateTitle(params),
  cancelTitleGen: () => browserChatRuntime.cancelTitleGen(),
  getSamplingDefaults: (repoId) => browserChatRuntime.samplingDefaults(repoId),
};

export const browserSession: SessionService = {
  onSessionState: (listener) => browserChatRuntime.onSessionState(listener),
  onSessionLoading: (listener) => browserChatRuntime.onSessionLoading(listener),
  ejectSession: ({ repoId }) => browserChatRuntime.eject(repoId),
};
