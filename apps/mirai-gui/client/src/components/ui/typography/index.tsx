import { createElement } from "react";
import { twMerge } from "tailwind-merge";
import { TEXT_COLOR, TEXT_SIZE } from "./constants";
import type { TextProps } from "./types";

export function Text({ size = "md", color = "primary", opticalSize, as = "p", className, style, ...rest }: TextProps) {
  return createElement(as, {
    className: twMerge(TEXT_SIZE[size], TEXT_COLOR[color], "font-normal", className),
    style: opticalSize ? { ...style, fontVariationSettings: `'opsz' ${opticalSize}` } : style,
    ...rest,
  });
}
