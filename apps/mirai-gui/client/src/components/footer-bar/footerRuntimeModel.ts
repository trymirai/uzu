import type { RuntimeSessionState } from "@/stores/useRuntimeSessionStore";
import type { RuntimeSessionRef } from "@/types/session";

type FooterModelInput = {
  repoId: string;
  name: string;
  vendor?: string;
};

type FooterModelLabel = {
  name: string;
  vendor: string;
};

type RuntimeSessions = Pick<RuntimeSessionState, "residentSession" | "loadingSession" | "ejectingSession">;

export type FooterRuntimeStatus = "hidden" | "loading" | "ejecting" | "ready";

export const getFooterModelLabel = (
  session: RuntimeSessionRef | null,
  chatModels: FooterModelInput[],
): FooterModelLabel => {
  if (!session) {
    return { name: "", vendor: "" };
  }

  const match = chatModels.find((model) => model.repoId === session.repoId);
  return {
    name: match?.name || session.repoId,
    vendor: match?.vendor || "",
  };
};

// Ejecting and loading outrank a resident session: both describe a transition
// away from it.
export const pickFooterActiveSession = (
  sessions: RuntimeSessions,
): {
  session: RuntimeSessionRef | null;
  status: FooterRuntimeStatus;
} => {
  if (sessions.ejectingSession) {
    return { session: sessions.ejectingSession, status: "ejecting" };
  }
  if (sessions.loadingSession) {
    return { session: sessions.loadingSession, status: "loading" };
  }
  if (sessions.residentSession) {
    return { session: sessions.residentSession, status: "ready" };
  }
  return { session: null, status: "hidden" };
};

export const canFooterEjectActiveSession = (
  sessions: RuntimeSessions,
  session: RuntimeSessionRef | null,
  status: FooterRuntimeStatus,
  ejectAwaiting: boolean,
  runtimeBusy: boolean,
): boolean => {
  if (!session || status !== "ready") {
    return false;
  }

  return (
    !!sessions.residentSession &&
    !sessions.loadingSession &&
    !sessions.ejectingSession &&
    !runtimeBusy &&
    !ejectAwaiting
  );
};
