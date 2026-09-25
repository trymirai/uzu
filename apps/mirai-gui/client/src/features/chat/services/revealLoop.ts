import { RevealPacer } from "./revealPacer";

type RevealLoopOptions = {
  baseText: string;
  /** False once another run has taken over; the loop then stops without applying anything. */
  isActive: () => boolean;
  apply: (visibleText: string) => void;
};

export type RevealLoop = {
  append: (delta: string) => void;
  /** Resolves once everything appended so far is on screen. */
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

  const schedule = (): void => {
    if (frame !== null) return;
    frame = window.requestAnimationFrame((now) => {
      frame = null;
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
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
        frame = null;
      }
      pacer.reset();
      releaseWaiters();
    },
  };
};
