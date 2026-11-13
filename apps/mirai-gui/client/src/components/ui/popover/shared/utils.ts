import type { PopoverAlign, PopoverSide } from "./types";

export const isPlainTextLabel = (value: unknown): value is string | number =>
  typeof value === "string" || typeof value === "number";

export const getOriginClassName = (side: PopoverSide, align: PopoverAlign) => {
  if (side === "top") {
    return align === "start" ? "origin-bottom-left" : "origin-bottom-right";
  }

  return align === "start" ? "origin-top-left" : "origin-top-right";
};
