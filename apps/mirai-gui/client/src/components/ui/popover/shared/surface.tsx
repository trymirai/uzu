import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { getAnimationClassName, getOriginClassName } from "./utils";
import type { PopoverAlign, PopoverSide } from "./types";

type PopoverSurfaceProps = {
  open: boolean;
  side: PopoverSide;
  align: PopoverAlign;
  className?: string;
  children?: ReactNode;
  animated?: boolean;
};

export function PopoverSurface({ open, side, align, className, children, animated = true }: PopoverSurfaceProps) {
  return (
    <div
      className={twMerge(
        "rounded-lg bg-surface-elevated shadow-sm p-1.5 min-w-[140px]",
        animated ? getOriginClassName(side, align) : undefined,
        animated ? getAnimationClassName(open) : undefined,
        className,
      )}
    >
      {children}
    </div>
  );
}
