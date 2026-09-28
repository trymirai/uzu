import type { ButtonSize, ButtonKind } from "./types";

export const SIZE_STYLES: Record<ButtonSize, string> = {
  xxs: "h-6 px-2 text-xs font-medium rounded gap-1",
  xs: "h-7 px-2.5 text-xs font-medium rounded-md gap-1.5",
  sm: "h-8 px-3 text-sm font-medium rounded-md gap-1.5",
  lg: "h-10 px-4 text-sm font-medium rounded-md gap-2",
};

export const ICON_ONLY_SIZE_STYLES: Record<ButtonSize, string> = {
  xxs: "size-6 rounded p-0",
  xs: "size-7 rounded-md p-0",
  sm: "size-8 rounded-md p-0",
  lg: "size-10 rounded-md p-0",
};

export const ICON_SIZE_MAP: Record<ButtonSize, number> = {
  xxs: 14,
  xs: 16,
  sm: 16,
  lg: 18,
};

export const KIND_STYLES: Record<ButtonKind, string> = {
  primary: "bg-primary text-primary-contrast hover:bg-primary-hover active:bg-primary-active active:scale-[0.97]",
  secondary:
    "bg-secondary text-secondary-contrast shadow-sm hover:bg-secondary-hover hover:shadow-md active:scale-[0.97]",
  danger: "bg-danger text-danger-contrast hover:bg-danger-hover active:bg-danger-active active:scale-[0.97]",
  ghost:
    "bg-transparent text-text-secondary hover:bg-secondary-hover hover:text-text-primary active:bg-secondary-active active:scale-[0.97]",
};

export const DISABLED_STYLES = "opacity-50 cursor-not-allowed pointer-events-none";
export const LOADING_STYLES = "cursor-wait pointer-events-none";
export const BASE_STYLES =
  "inline-flex shrink-0 items-center justify-center whitespace-nowrap select-none outline-hidden antialiased transition-all duration-150 ease-out will-change-transform touch-manipulation focus-visible:shadow-focus [&_svg]:pointer-events-none [&_svg]:shrink-0";
export const FULL_WIDTH_STYLES = "w-full";

export const SPINNER_SIZE_MAP: Record<ButtonSize, number> = {
  xxs: 10,
  xs: 12,
  sm: 14,
  lg: 16,
};
