import type { SessionService } from ".";
import { noopUnsubscribe } from "../shared/noop";

export const webSession: SessionService = {
  onSessionState: noopUnsubscribe,
  onSessionLoading: noopUnsubscribe,
  ejectSession: () => Promise.reject(new Error("Not available on web")),
};
