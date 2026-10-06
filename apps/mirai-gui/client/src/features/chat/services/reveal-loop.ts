import { RevealPacer } from "./reveal-pacer";

type RevealLoopOptions = {
  baseText: string;
  /** False once another run has taken over; the loop then stops without applying anything. */
  isActive: () => boolean;
  apply: (visibleText: string) => void;
};

export type RevealLoop = {
  append: (delta: string) => void;
  /** Resolves once everything appended so far is on screen, or once the loop is cancelled. */
  drain: () => Promise<void>;
  cancel: () => void;
};

// Drives a RevealPacer from requestAnimationFrame, so received text reaches the
// store once per frame instead of once per chunk.
export const createRevealLoop = ({ baseText, isActive, apply }: RevealLoopOptions): RevealLoop => {
  const pacer = new RevealPacer(baseText);
  let lastFrameAt = 0;
  let frame: number | null = null;
  let waiters: Array<() => void> = [];

  const releaseWaiters = (): void => {
    const pending = waiters;
    waiters = [];
    pending.forEach((resolve) => resolve());
  };

  // A hidden window gets no animation frames, even one already requested.
  const onVisibilityChange = (): void => {
    if (document.visibilityState === "hidden") schedule();
  };
  const cancelFrame = (): void => {
    if (frame === null) return;
    window.cancelAnimationFrame(frame);
    frame = null;
    document.removeEventListener("visibilitychange", onVisibilityChange);
  };

  const schedule = (): void => {
    if (document.visibilityState === "hidden") {
      cancelFrame();
      if (isActive()) {
        const visible = pacer.flush();
        if (visible !== null) apply(visible);
      }
      releaseWaiters();
      return;
    }
    if (frame !== null) return;
    document.addEventListener("visibilitychange", onVisibilityChange);
    frame = window.requestAnimationFrame((now) => {
      frame = null;
      document.removeEventListener("visibilitychange", onVisibilityChange);
      if (!isActive()) {
        releaseWaiters();
        return;
      }
      const dtMs = lastFrameAt === 0 ? 16 : Math.min(now - lastFrameAt, 100);
      lastFrameAt = now;
      const visible = pacer.step(dtMs);
      if (visible !== null) apply(visible);
      if (pacer.isCaughtUp) {
        releaseWaiters();
      } else {
        schedule();
      }
    });
  };

  return {
    append: (delta) => {
      pacer.append(delta);
      schedule();
    },
    drain: () => {
      pacer.markDone();
      if (pacer.isCaughtUp && frame === null) return Promise.resolve();
      return new Promise((resolve) => {
        waiters.push(resolve);
        schedule();
      });
    },
    cancel: () => {
      cancelFrame();
      pacer.reset();
      releaseWaiters();
    },
  };
};
