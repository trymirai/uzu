import { twMerge } from "tailwind-merge";
import { Dialog, DialogPanel, DialogTitle } from "@headlessui/react";
import { X } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { UzuCat, UzuCatStill } from "@/components/uzu-cat";

/*
 * Uzu bird: a tiny flappy game to pass a long think. The cat is the bird; the pipes are drawn like
 * its brackets. Space, click or tap to flap.
 */

const W = 360;
const H = 360;
const CAT = 40;
const CAT_X = 72;
const HIT = 24; // forgiving hitbox inside the drawing
const GRAVITY = 1500; // px/s²
const FLAP = -430; // px/s
const SPEED = 140; // px/s
const PIPE_W = 52;
const GAP = 138;
const SPACING = 190;
const BEST_KEY = "mirai.uzuBird.best";

type Pipe = { x: number; gapY: number; passed: boolean };
type Phase = "playing" | "over";

const readBest = () => {
  try {
    return Number(localStorage.getItem(BEST_KEY)) || 0;
  } catch {
    return 0;
  }
};
const saveBest = (best: number) => {
  try {
    localStorage.setItem(BEST_KEY, String(best));
  } catch {
    // Private windows and blocked storage: the best score lasts for this session only.
  }
};

const newPipe = (x: number): Pipe => ({ x, gapY: 90 + Math.random() * (H - 180 - GAP), passed: false });

function Game({
  answerReady,
  onBack,
  jumpRef,
}: {
  answerReady: boolean;
  onBack: () => void;
  /** Filled with the flap action, so the "Space to jump" button outside the field can use it. */
  jumpRef: { current: (() => void) | null };
}) {
  // It starts the moment the window opens, already on its way up.
  const [phase, setPhase] = useState<Phase>("playing");
  const [score, setScore] = useState(0);
  const [best, setBest] = useState(readBest);
  const [, setFrame] = useState(0);
  const world = useRef({ y: H / 2 - CAT / 2, vy: FLAP, pipes: [newPipe(W + 40), newPipe(W + 40 + SPACING)] });
  const flappedAt = useRef(performance.now());
  const phaseRef = useRef<Phase>("playing");
  phaseRef.current = phase;

  const reset = useCallback(() => {
    world.current = { y: H / 2 - CAT / 2, vy: FLAP, pipes: [newPipe(W + 40), newPipe(W + 40 + SPACING)] };
    flappedAt.current = performance.now();
    setScore(0);
    setPhase("playing");
  }, []);

  const flap = useCallback(() => {
    if (phaseRef.current === "over") return;
    world.current.vy = FLAP;
    flappedAt.current = performance.now();
  }, []);

  useEffect(() => {
    if (phase !== "playing") return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const w = world.current;
      w.vy += GRAVITY * dt;
      w.y += w.vy * dt;
      for (const pipe of w.pipes) pipe.x -= SPEED * dt;
      if (w.pipes[0]!.x < -PIPE_W) {
        w.pipes.shift();
        w.pipes.push(newPipe(w.pipes[w.pipes.length - 1]!.x + SPACING));
      }
      const cx = CAT_X + CAT / 2;
      const cy = w.y + CAT / 2;
      let crashed = cy - HIT / 2 < 0 || cy + HIT / 2 > H;
      for (const pipe of w.pipes) {
        const overlapsX = cx + HIT / 2 > pipe.x && cx - HIT / 2 < pipe.x + PIPE_W;
        if (overlapsX && (cy - HIT / 2 < pipe.gapY || cy + HIT / 2 > pipe.gapY + GAP)) crashed = true;
        if (!pipe.passed && pipe.x + PIPE_W < cx) {
          pipe.passed = true;
          setScore((s) => s + 1);
        }
      }
      if (crashed) {
        setPhase("over");
        return;
      }
      setFrame((f) => f + 1);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [phase]);

  useEffect(() => {
    if (phase !== "over" || score <= best) return;
    setBest(score);
    saveBest(score);
  }, [phase, score, best]);

  useEffect(() => {
    jumpRef.current = () => (phaseRef.current === "over" ? reset() : flap());
    return () => {
      jumpRef.current = null;
    };
  }, [jumpRef, flap, reset]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" && e.code !== "ArrowUp") return;
      e.preventDefault();
      if (e.repeat) return;
      if (phaseRef.current === "over") reset();
      else flap();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flap, reset]);

  const w = world.current;
  const justFlapped = performance.now() - flappedAt.current < 220;

  const tilt = phase === "playing" ? Math.max(-25, Math.min(60, w.vy / 9)) : 0;

  return (
    <div
      role="application"
      aria-label="Uzu bird game"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        if (phase === "over") return;
        flap();
      }}
      className="relative overflow-hidden rounded-xl bg-surface-soft select-none touch-none"
      style={{ width: W, height: H }}
    >
      {w.pipes.map((pipe, i) => (
        <div key={i} aria-hidden>
          <div
            className="absolute rounded-b-[14px] border-[5px] border-t-0 border-text-primary"
            style={{ left: pipe.x, top: -6, width: PIPE_W, height: pipe.gapY + 6 }}
          />
          <div
            className="absolute rounded-t-[14px] border-[5px] border-b-0 border-text-primary"
            style={{ left: pipe.x, top: pipe.gapY + GAP, width: PIPE_W, height: H - pipe.gapY - GAP + 6 }}
          />
        </div>
      ))}

      <div
        className="absolute text-text-primary"
        style={{ left: CAT_X, top: w.y, width: CAT, height: CAT, transform: `rotate(${tilt}deg)` }}
      >
        {/* Flying, the cat holds still with its normal eyes (no blinking), pleased on each flap;
            only a crash brings the live, trembling face. */}
        {phase === "over" ? (
          <UzuCat state="memory" size={CAT} />
        ) : (
          <UzuCatStill pose={justFlapped ? "happy" : "default"} size={CAT} />
        )}
      </div>

      <div className="absolute inset-x-0 top-4 text-center text-[28px] font-[600] tabular-nums text-text-primary">
        {score}
      </div>

      {phase === "over" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-surface-soft/70 backdrop-blur-[2px] animate-uzu-in">
          <p className="text-[15px] font-[500] text-label-title">Score {score}</p>
          <p className="text-[13px] text-label-muted tabular-nums">Best {Math.max(best, score)}</p>
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={reset}
            className="mt-4 h-8 rounded-full bg-primary px-4 text-[13px] font-[500] text-primary-contrast outline-hidden transition-[scale] duration-150 ease-out active:scale-[0.97] focus-visible:shadow-focus"
          >
            Play again
          </button>
        </div>
      ) : null}

      {answerReady ? (
        <div className="absolute inset-x-3 bottom-3 flex items-center justify-between gap-2 rounded-xl bg-surface-popover px-3 py-2 shadow-popover animate-uzu-in">
          <span className="text-[13px] font-[450] text-label-title">Your answer is ready</span>
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={onBack}
            className="flex h-7 items-center gap-1.5 rounded-full bg-primary pl-3 pr-1 text-[12px] font-[500] text-primary-contrast outline-hidden focus-visible:shadow-focus"
          >
            Back to chat
            <kbd className="rounded-full bg-primary-contrast/15 px-2 py-0.5 font-sans text-[11px] font-[500]">Esc</kbd>
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** A key hint that is also the action, for pointer users. */
function HintButton({
  keyLabel,
  onClick,
  children,
  className,
}: {
  keyLabel: string;
  onClick: () => void;
  children: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={twMerge(
        "flex h-7 items-center gap-1.5 rounded-full bg-control-fill pl-1 pr-3 text-[12px] font-[450] text-text-secondary outline-hidden transition-colors duration-150 ease-out hover:bg-control-fill-hover hover:text-label-title focus-visible:shadow-focus",
        className,
      )}
    >
      <kbd className="rounded-full bg-control-selected px-2 py-0.5 font-sans text-[11px] font-[500] text-label-title shadow-[0_0_0_0.5px_rgb(0_0_0/0.08)] dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)]">
        {keyLabel}
      </kbd>
      {children}
    </button>
  );
}

/** The game window. */
export function UzuBirdDialog({
  open,
  onClose,
  answerReady,
}: {
  open: boolean;
  onClose: () => void;
  answerReady: boolean;
}) {
  const jumpRef = useRef<(() => void) | null>(null);
  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/20 backdrop-blur-[6px]" aria-hidden />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="rounded-2xl bg-surface-popover p-3 shadow-popover animate-uzu-in">
          <div className="flex items-center justify-between px-1 pb-3">
            <DialogTitle className="text-[15px] font-[600] text-label-title">Uzu bird</DialogTitle>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="flex size-7 items-center justify-center rounded-full text-label-muted outline-hidden transition-colors duration-150 ease-out hover:bg-surface-tertiary hover:text-label-title focus-visible:shadow-focus"
            >
              <X className="size-4" />
            </button>
          </div>
          <Game answerReady={answerReady} onBack={onClose} jumpRef={jumpRef} />
          <div className="flex items-center justify-center gap-2 pt-3">
            <HintButton keyLabel="Space" onClick={() => jumpRef.current?.()}>
              to jump
            </HintButton>
            <HintButton keyLabel="Esc" onClick={onClose}>
              to close
            </HintButton>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}

/*
 * Whether the game is open lives outside the message that offered it. When the answer lands, the
 * in-progress message is replaced by the finished one; the game must outlive that swap, so the
 * window is drawn by UzuBirdHost in the message list and only told here that the answer is ready.
 */
type BirdState = { open: boolean; answerReady: boolean };
let birdState: BirdState = { open: false, answerReady: false };
const birdListeners = new Set<() => void>();
function setBird(patch: Partial<BirdState>) {
  const next = { ...birdState, ...patch };
  if (next.open === birdState.open && next.answerReady === birdState.answerReady) return;
  birdState = next;
  birdListeners.forEach((listener) => listener());
}
function subscribeBird(listener: () => void) {
  birdListeners.add(listener);
  return () => birdListeners.delete(listener);
}
const getBird = () => birdState;
const useBird = () => useSyncExternalStore(subscribeBird, getBird, getBird);

/** Tells the game whether the newest answer has landed, so it can offer the way back. */
export function UzuBirdAnswerSync({ answerReady }: { answerReady: boolean }) {
  useEffect(() => {
    setBird({ answerReady });
  }, [answerReady]);
  return null;
}

const TAPS = 2;
const TAP_WINDOW_MS = 500;

/**
 * The way in is hidden: a double tap on a cat beside an answer. Returns handlers for the cat's
 * box. Opening clears "answer ready", so a game started after the answer only offers the way back
 * once a new answer lands.
 */
export function useUzuBirdTaps() {
  const taps = useRef<number[]>([]);
  return {
    // Repeated clicks would otherwise select the text beside the cat.
    onMouseDown: (e: React.MouseEvent) => {
      if (e.detail > 1) e.preventDefault();
    },
    onClick: () => {
      const now = performance.now();
      taps.current = [...taps.current.filter((t) => now - t < TAP_WINDOW_MS), now];
      if (taps.current.length < TAPS) return;
      taps.current = [];
      setBird({ open: true, answerReady: false });
    },
  };
}

/** The game window, mounted once with the message list so it stays open while the answer lands. */
export function UzuBirdHost() {
  const { open, answerReady } = useBird();
  // Leaving the chat closes the game rather than leaving it waiting for the next one.
  useEffect(() => () => setBird({ open: false }), []);
  return <UzuBirdDialog open={open} onClose={() => setBird({ open: false })} answerReady={answerReady} />;
}
