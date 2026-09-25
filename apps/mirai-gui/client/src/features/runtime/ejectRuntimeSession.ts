import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import { isRuntimeBusy } from "@/features/runtime/runtimeBusy";
import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { runtimeSessionEjectReasons, type RuntimeSessionEjectReason, type RuntimeSessionRef } from "@/types/session";
import { getPlatform } from "@/platform/platformSingleton";

type EjectRuntimeSessionOptions = {
  target: RuntimeSessionRef;
  reason?: RuntimeSessionEjectReason;
};

const isTargetGone = (current: RuntimeSessionRef | null, target: RuntimeSessionRef): boolean =>
  !current || current.repoId !== target.repoId;

const waitForRuntimeSessionEject = (target: RuntimeSessionRef, abandoned: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    const current = useRuntimeSessionStore.getState();
    const alreadyGone = isTargetGone(current.residentSession, target);
    const currentlyEjectingTarget = !isTargetGone(current.ejectingSession, target);
    if (alreadyGone && !currentlyEjectingTarget) {
      resolve();
      return;
    }

    const unsubscribe = useRuntimeSessionStore.subscribe((state) => {
      const targetGone = isTargetGone(state.residentSession, target);
      const isEjectingTarget = !isTargetGone(state.ejectingSession, target);
      if (targetGone && !isEjectingTarget) {
        unsubscribe();
        resolve();
      }
    });
    abandoned.addEventListener("abort", unsubscribe, { once: true });
  });

// Subscribes before the request so an eject that completes first is not missed.
const requestRuntimeSessionEject = async (target: RuntimeSessionRef): Promise<void> => {
  const abandon = new AbortController();
  const ejected = waitForRuntimeSessionEject(target, abandon.signal);
  try {
    await getPlatform().session.ejectSession(toRuntimeEjectPayload(target));
  } catch (error) {
    abandon.abort();
    throw error;
  }
  await ejected;
};

const toRuntimeEjectPayload = (target: RuntimeSessionRef) => ({ repoId: target.repoId });

export const ejectRuntimeSessionAndWait = async ({
  target,
  reason = runtimeSessionEjectReasons.auto,
}: EjectRuntimeSessionOptions): Promise<boolean> => {
  if (isRuntimeBusy()) {
    throw new Error("Cannot eject a model while generation is in progress");
  }

  await requestRuntimeSessionEject(target);
  useChatSessionStore.getState().setLastEjectedWithReason(target, reason);

  return true;
};

export const ejectAndWait = async (
  target: RuntimeSessionRef,
  reason: RuntimeSessionEjectReason = runtimeSessionEjectReasons.auto,
): Promise<boolean> => {
  const result = await useChatSessionStore
    .getState()
    .withOperation("ejecting", () => ejectRuntimeSessionAndWait({ target, reason }));
  return result === true;
};
