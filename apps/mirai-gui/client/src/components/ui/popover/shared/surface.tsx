import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { getOriginClassName } from "./utils";
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
        "rounded-xl bg-surface-popover shadow-popover p-1 min-w-[180px]",
        // The panel mounts already open, so the entrance is a keyframe rather than a transition.
        animated ? getOriginClassName(side, align) : undefined,
        animated && open ? "animate-uzu-pop" : undefined,
        className,
      )}
    >
      {children}
    </div>
  );
}
