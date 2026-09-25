import { listen } from "@tauri-apps/api/event";
import { invoke } from "../shared/invoke";
import type { SessionLoadingPayload, SessionService, SessionStatePayload } from ".";

const stateListeners = new Set<(p: SessionStatePayload) => void>();
const loadingListeners = new Set<(p: SessionLoadingPayload) => void>();
let listening = false;

const fanOut = <T>(set: Set<(p: T) => void>, payload: T): void => {
  set.forEach((cb) => {
    try {
      cb(payload);
    } catch (error) {
      // The other listeners still get the event.
      console.error("[session] listener failed", error);
    }
  });
};

const ensureListening = (): void => {
  if (listening) return;
  listening = true;
  void listen<SessionStatePayload>("session-state", (event) => fanOut(stateListeners, event.payload));
  void listen<SessionLoadingPayload>("session-loading", (event) => fanOut(loadingListeners, event.payload));
};

const subscribe = <T>(set: Set<(p: T) => void>, cb: (p: T) => void): (() => void) => {
  ensureListening();
  set.add(cb);
  return () => {
    set.delete(cb);
  };
};

export const tauriSession: SessionService = {
  onSessionState: (cb) => subscribe(stateListeners, cb),
  onSessionLoading: (cb) => subscribe(loadingListeners, cb),
  ejectSession: ({ repoId }) => invoke("eject_session", { repoId }),
};
