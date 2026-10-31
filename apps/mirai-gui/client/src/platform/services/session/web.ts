import type { SessionService } from ".";
import { noopUnsubscribe } from "../shared/noop";

export const webSession: SessionService = {
  onSessionState: noopUnsubscribe,
  onSessionLoading: noopUnsubscribe,
  getResidentSession: () => Promise.resolve(null),
  ejectSession: () => Promise.reject(new Error("Not available on web")),
};
