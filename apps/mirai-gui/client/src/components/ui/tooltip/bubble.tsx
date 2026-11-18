import type { CSSProperties } from "react";
import { twMerge } from "tailwind-merge";
import type { TooltipProps } from "./types";

type TooltipBubbleProps = {
  tooltipId: string;
  side: NonNullable<TooltipProps["side"]>;
  align: NonNullable<TooltipProps["align"]>;
  variant: NonNullable<TooltipProps["variant"]>;
  instant: boolean;
  content: TooltipProps["content"];
  floatingStyles: CSSProperties;
  positioned: boolean;
  setFloating: (node: HTMLDivElement | null) => void;
  onPointerEnter: () => void;
  onPointerLeave: (event: React.PointerEvent) => void;
};

export function TooltipBubble({
  tooltipId,
  side,
  align,
  variant,
  instant,
  content,
  floatingStyles,
  positioned,
  setFloating,
  onPointerEnter,
  onPointerLeave,
}: TooltipBubbleProps) {
  return (
    <div
      ref={setFloating}
      id={tooltipId}
      role="tooltip"
      // A label tip lets the pointer through; a panel can be hovered, and its figures selected.
      className={twMerge("fixed z-50", variant === "panel" ? "select-text" : "pointer-events-none")}
      onPointerEnter={variant === "panel" ? onPointerEnter : undefined}
      onPointerLeave={variant === "panel" ? onPointerLeave : undefined}
      style={{
        ...floatingStyles,
        // Hidden until it has been placed against its trigger, so it can't flash somewhere else.
        visibility: positioned ? "visible" : "hidden",
      }}
    >
      {/* The same floating surface as menus, so it matches the theme it appears in. */}
      <div
        data-side={side}
        className={twMerge(
          "bg-surface-popover text-text-primary shadow-popover whitespace-nowrap tabular-nums",
          // A panel matches the hover menus (12px corners, text 12px in from the edge); a label is a
          // small tip.
          variant === "panel"
            ? "rounded-xl px-3 py-2.5"
            : "rounded-lg px-2.5 py-1.5 text-[12px] font-[500] leading-[16px]",
          // Grows from the trigger: its middle, or its corner when lined up with its edge.
          side === "top"
            ? align === "start"
              ? "origin-bottom-left"
              : "origin-bottom"
            : align === "start"
              ? "origin-top-left"
              : "origin-top",
          instant ? undefined : side === "top" ? "animate-uzu-tip-up" : "animate-uzu-tip-down",
        )}
      >
        {content}
      </div>
    </div>
  );
}
