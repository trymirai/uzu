import type { SessionService } from ".";
import { browserRuntime } from "../chat/browser-runtime";

export const webSession: SessionService = {
  onSessionState: (listener) => browserRuntime.onSessionState(listener),
  onSessionLoading: (listener) => browserRuntime.onSessionLoading(listener),
  ejectSession: ({ repoId }) => browserRuntime.eject(repoId),
};
