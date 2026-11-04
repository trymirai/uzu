import type { CSSProperties } from "react";
import { twMerge } from "tailwind-merge";
import type { TooltipProps } from "./types";

type TooltipBubbleProps = {
  tooltipId: string;
  side: NonNullable<TooltipProps["side"]>;
  instant: boolean;
  content: TooltipProps["content"];
  floatingStyles: CSSProperties;
  setFloating: (node: HTMLDivElement | null) => void;
};

export function TooltipBubble({ tooltipId, side, instant, content, floatingStyles, setFloating }: TooltipBubbleProps) {
  return (
    <div
      ref={setFloating}
      id={tooltipId}
      role="tooltip"
      className="pointer-events-none fixed z-50"
      style={{
        ...floatingStyles,
        opacity: 1,
      }}
    >
      {/* The same floating surface as menus, so it matches the theme it appears in. */}
      <div
        data-side={side}
        className={twMerge(
          "rounded-lg bg-surface-popover text-text-primary shadow-popover",
          "px-2.5 py-1.5 text-[12px] font-[500] leading-[16px] tabular-nums",
          "whitespace-nowrap",
          side === "top" ? "origin-bottom" : "origin-top",
          instant ? undefined : side === "top" ? "animate-uzu-tip-up" : "animate-uzu-tip-down",
        )}
      >
        {content}
      </div>
    </div>
  );
}
