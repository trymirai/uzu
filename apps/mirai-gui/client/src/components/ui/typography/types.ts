import type { HTMLAttributes } from "react";

export type TextSize = "sm" | "md";

export type TextColor = "primary" | "secondary" | "muted";

export type TextProps = HTMLAttributes<HTMLParagraphElement> & {
  size?: TextSize;
  color?: TextColor;
  opticalSize?: number;
  as?: "p" | "span";
  className?: string;
};
