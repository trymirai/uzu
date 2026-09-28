import type { SelectorSize } from "./types";

export const CHECKBOX_SIZE: Record<SelectorSize, { box: string; icon: number; radius: string }> = {
  sm: { box: "size-4", icon: 12, radius: "rounded" },
  md: { box: "size-[18px]", icon: 14, radius: "rounded" },
};

export const CHECKED_BG = "bg-primary";

export const CHECKED_BORDER = "border-primary";

export const UNCHECKED_BORDER = "border-border-strong";

export const FOCUS_RING = "focus-visible:shadow-focus";
