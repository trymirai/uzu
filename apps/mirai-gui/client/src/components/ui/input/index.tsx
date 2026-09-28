import { forwardRef } from "react";
import { twMerge } from "tailwind-merge";
import {
  DISABLED_INPUT,
  DISABLED_WRAPPER,
  FULL_WIDTH,
  ICON_COLOR,
  INPUT_BASE,
  KIND_BORDER,
  KIND_FOCUS_RING,
  KIND_HOVER,
  INPUT_SIZE_STYLES,
  WRAPPER_BASE,
} from "./constants";
import type { InputProps } from "./types";
import { applyIconSize } from "./utils";

export type { InputProps } from "./types";

/** Low-level input control for inline UIs. For forms, use TextField. */
export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(props, ref) {
  const {
    size = "md",
    kind = "default",
    leftIcon,
    fullWidth = false,
    disabled = false,
    className,
    ...inputProps
  } = props;

  const sizeConfig = INPUT_SIZE_STYLES[size];

  const wrapperClasses = twMerge(
    WRAPPER_BASE,
    sizeConfig.wrapper,
    KIND_BORDER[kind],
    !disabled ? KIND_HOVER[kind] : "",
    !disabled ? KIND_FOCUS_RING[kind] : "",
    disabled ? DISABLED_WRAPPER : "",
    fullWidth ? FULL_WIDTH : "",
    className,
  );

  const inputClasses = twMerge(INPUT_BASE, sizeConfig.input, disabled ? DISABLED_INPUT : "");

  const iconColor = ICON_COLOR[kind];
  const leftIconNode = applyIconSize(leftIcon, sizeConfig.icon);

  return (
    <div className={wrapperClasses}>
      {leftIconNode && (
        <span className={twMerge("shrink-0 flex items-center justify-center", iconColor)} aria-hidden="true">
          {leftIconNode}
        </span>
      )}
      <input
        ref={ref}
        {...inputProps}
        disabled={disabled}
        className={inputClasses}
        aria-invalid={kind === "error" || undefined}
      />
    </div>
  );
});
