import { createElement, isValidElement, type ReactNode, type ElementType } from "react";
import type { ButtonKind, ButtonSize } from "./types";
import { ICON_SIZE_MAP } from "./constants";

export function getButtonIcon(icon: ElementType | ReactNode | undefined, size: ButtonSize): ReactNode {
  if (!icon) return null;
  const iconSize = ICON_SIZE_MAP[size];
  const iconStyle = { width: iconSize, height: iconSize };
  if (isValidElement(icon)) return icon;
  // forwardRef and memo components (every lucide icon) are objects, not functions.
  if (typeof icon !== "function" && typeof icon !== "object") return null;
  return createElement(icon as ElementType, { style: iconStyle, "aria-hidden": true });
}

type ButtonCommonPropsParams = {
  className: string;
  ariaLabel: string | undefined;
  disabled: boolean;
  loading: boolean;
  size: ButtonSize;
  kind: ButtonKind;
  isIconOnly: boolean;
};

export function getButtonCommonProps({
  className,
  ariaLabel,
  disabled,
  loading,
  size,
  kind,
  isIconOnly,
}: ButtonCommonPropsParams) {
  return {
    className,
    "aria-label": ariaLabel,
    "aria-disabled": disabled || loading,
    "aria-busy": loading,
    "data-size": size,
    "data-kind": kind,
    "data-loading": loading || undefined,
    "data-icon-only": isIconOnly || undefined,
  };
}
