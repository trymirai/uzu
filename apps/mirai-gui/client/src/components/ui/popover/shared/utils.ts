import { twMerge } from "tailwind-merge";
import { TRANSITION_BASE } from "./constants";
import type { PopoverAlign, PopoverSide } from "./types";

export const isPlainTextLabel = (value: unknown): value is string | number =>
  typeof value === "string" || typeof value === "number";

export const getOriginClassName = (side: PopoverSide, align: PopoverAlign) => {
  if (side === "top") {
    return align === "start" ? "origin-bottom-left" : "origin-bottom-right";
  }

  return align === "start" ? "origin-top-left" : "origin-top-right";
};

export const getAnimationClassName = (open: boolean) =>
  twMerge(
    TRANSITION_BASE,
    open ? "duration-[120ms]" : "duration-[80ms]",
    open ? "opacity-100 scale-100" : "opacity-0 scale-90",
  );
