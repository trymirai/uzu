import type { TextColor, TextSize } from "./types";

export const TEXT_SIZE: Record<TextSize, string> = {
  sm: "text-sm leading-[1.55]",
  md: "text-base leading-[1.6]",
};

export const TEXT_COLOR: Record<TextColor, string> = {
  primary: "text-text-primary",
  secondary: "text-text-secondary",
  muted: "text-text-muted",
};
