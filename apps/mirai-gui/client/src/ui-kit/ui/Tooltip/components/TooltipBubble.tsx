import type { CSSProperties } from "react";
import { twMerge } from "tailwind-merge";
import type { TooltipProps } from "../types";

type TooltipBubbleProps = {
  tooltipId: string;
  content: TooltipProps["content"];
  floatingStyles: CSSProperties;
  setFloating: (node: HTMLDivElement | null) => void;
};

export function TooltipBubble({ tooltipId, content, floatingStyles, setFloating }: TooltipBubbleProps) {
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
      <div
        className={twMerge(
          "rounded-md bg-surface-tertiary shadow-sm text-text-primary [font-variation-settings:'opsz'_18]",
          "px-2 py-1 text-[11px] font-[450] leading-[1.3]",
          "whitespace-nowrap",
        )}
      >
        {content}
      </div>
    </div>
  );
}
