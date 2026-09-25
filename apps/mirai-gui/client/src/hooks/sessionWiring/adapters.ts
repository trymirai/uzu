import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import { useChatSessionStore } from "@/stores/useChatSessionStore";
import type { RuntimeSessionRef } from "@/types/session";
import { getPlatform } from "@/platform/platformSingleton";
import type { SessionLoadingPayload, SessionStatePayload } from "@/platform/services/session";

const toChatSession = (payload: { repoId?: string; active?: boolean }): RuntimeSessionRef | null => {
  if (!payload.repoId || payload.active === false) {
    return null;
  }
  return { repoId: payload.repoId };
};

const toChatSessionTarget = (payload: { repoId?: string }): RuntimeSessionRef | null =>
  payload.repoId ? { repoId: payload.repoId } : null;

const applyChatSessionState = (payload: SessionStatePayload, session: RuntimeSessionRef | null): void => {
  const target = toChatSessionTarget(payload);
  if (!target) {
    return;
  }

  const currentResident = useRuntimeSessionStore.getState().residentSession;
  const currentEjecting = useRuntimeSessionStore.getState().ejectingSession;
  const isCurrentResidentTarget = currentResident?.repoId === target.repoId;
  const isCurrentEjectingTarget = currentEjecting?.repoId === target.repoId;

  if (payload.isEjecting === true) {
    if (isCurrentResidentTarget || isCurrentEjectingTarget) {
      useChatSessionStore.getState().setEjecting(true);
      useRuntimeSessionStore.getState().setEjectingSession(target);
    }
  }

  if (payload.active === true) {
    useChatSessionStore.getState().endModelLoading();
    useRuntimeSessionStore.getState().setResidentSession(session);
  }

  if (payload.active === false && isCurrentResidentTarget) {
    useRuntimeSessionStore.getState().setResidentSession(null);
  }

  if (payload.isEjecting === false && (isCurrentEjectingTarget || isCurrentResidentTarget)) {
    useChatSessionStore.getState().setEjecting(false);
    useRuntimeSessionStore.getState().setEjectingSession(null);
  }
};

const applyChatLoading = (payload: SessionLoadingPayload): void => {
  if (payload.status === "start") {
    useChatSessionStore.getState().startModelLoading();
    useRuntimeSessionStore.getState().startLoadingSession({ repoId: payload.repoId });
  }
  if (payload.status === "error") {
    useChatSessionStore.getState().endModelLoading();
    useRuntimeSessionStore.getState().endLoadingSession();
  }
};

export const bindChatSessionState = (): (() => void) => {
  const { session } = getPlatform();
  return session.onSessionState((payload) => {
    applyChatSessionState(payload, toChatSession(payload));
  });
};

export const bindChatLoading = (): (() => void) => getPlatform().session.onSessionLoading(applyChatLoading);
