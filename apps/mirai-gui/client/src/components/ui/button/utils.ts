import type { ButtonKind, ButtonSize } from "./types";

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
