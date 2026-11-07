import { type RefObject, useId, useRef } from "react";
import { twMerge } from "tailwind-merge";
import { useAppStore } from "@/stores/use-app-store";
import { drawFace } from "./face";
import { type Face, type StateId, useFace } from "./engine";
import { type PoseId, poseFace } from "./poses";
import { FAVOURITE, forSize } from "./settings";

export type { StateId as UzuCatState } from "./engine";

/** The app background in each theme: empty loader eyes are filled with what the cat sits on. */
const PAPER = { light: "#fcfcfc", dark: "#1c1c1c" } as const;

type DrawProps = {
  face: Face;
  size: number;
  /** The surface under the cat, when it is not the app background (a card, the composer). */
  paper?: string;
  label?: string;
  className?: string;
};

function CatSvg({ face, size, paper, label, className }: DrawProps) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const isDarkMode = useAppStore((s) => s.isDarkMode);
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 160 160"
      fill="none"
      className={twMerge("shrink-0 select-none pointer-events-none", className)}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {drawFace({
        f: face,
        s: forSize(FAVOURITE, size),
        ink: "currentColor",
        paper: paper ?? PAPER[isDarkMode ? "dark" : "light"],
        uid,
      })}
    </svg>
  );
}

type UzuCatProps = Omit<DrawProps, "face"> & {
  state: StateId;
  /** For "watch": where to look, -1..1 each way from the cat (x right, y down). */
  lookAt?: { x: number; y: number } | null;
  /** For "loading": real progress 0..1. */
  progress?: number | null;
  /** The element the cat is measured from (for following the pointer); defaults to its own box. */
  anchorRef?: RefObject<HTMLElement | null>;
};

/** The live uzu cat: blinks, glances and moves through its states. */
export function UzuCat({ state, lookAt, progress, anchorRef, size, ...draw }: UzuCatProps) {
  const ownRef = useRef<HTMLSpanElement>(null);
  const face = useFace(
    {
      state,
      blinkRate: FAVOURITE.blinkRate,
      gazeRange: FAVOURITE.gazeRange,
      speed: FAVOURITE.speed,
      restartKey: 0,
      lookAt,
      progress,
    },
    anchorRef ?? ownRef,
  );
  return (
    <span ref={ownRef} className="inline-flex shrink-0">
      <CatSvg face={face} size={size} {...draw} />
    </span>
  );
}

/** The cat holding one still pose, for places that should not move (history, icons). */
export function UzuCatStill({ pose = "default", ...draw }: Omit<DrawProps, "face"> & { pose?: PoseId }) {
  return <CatSvg face={poseFace(pose)} {...draw} />;
}
