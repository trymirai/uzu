import type { InputSize, InputKind } from "./types";

export const WRAPPER_BASE =
  "inline-flex items-center bg-surface-secondary border-[0.5px] rounded-md transition-all duration-150 ease-out";

export const INPUT_SIZE_STYLES: Record<InputSize, { wrapper: string; input: string; icon: number }> = {
  sm: { wrapper: "h-8 px-2.5 gap-1.5", input: "text-xs", icon: 14 },
  md: { wrapper: "h-9 px-3 gap-2", input: "text-sm", icon: 16 },
};

export const KIND_BORDER: Record<InputKind, string> = {
  default: "border-border-outlined",
  error: "border-red-500",
};

// Focus is shown with the border alone: an outward ring or ring-offset overflows
// the wrapper and collides with neighbours in tight rows (model params, toolbars).
export const KIND_FOCUS_RING: Record<InputKind, string> = {
  default: "focus-within:border-border-outlined-hover",
  error: "focus-within:border-red-500",
};

export const KIND_HOVER: Record<InputKind, string> = {
  default: "hover:border-border-outlined-hover",
  error: "hover:border-red-500",
};

export const INPUT_BASE =
  "flex-1 min-w-0 bg-transparent outline-none text-text-primary placeholder:text-text-muted font-normal";

export const DISABLED_WRAPPER = "opacity-40 cursor-not-allowed";
export const DISABLED_INPUT = "cursor-not-allowed";

export const ICON_COLOR: Record<InputKind, string> = {
  default: "text-text-muted",
  error: "text-red-500",
};

export const FULL_WIDTH = "w-full";
