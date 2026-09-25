import { forwardRef } from "react";
import { twMerge } from "tailwind-merge";
import { Spinner } from "../Spinner";
import {
  BASE_STYLES,
  DISABLED_STYLES,
  FULL_WIDTH_STYLES,
  ICON_ONLY_SIZE_STYLES,
  KIND_STYLES,
  LOADING_STYLES,
  SIZE_STYLES,
  SPINNER_SIZE_MAP,
} from "./constants";
import type { ButtonAsButton, ButtonAsLink, ButtonProps } from "./types";
import { getButtonCommonProps, getButtonIcon } from "./utils";

type SharedButtonPropKeys =
  | "children"
  | "size"
  | "kind"
  | "icon"
  | "iconPosition"
  | "loading"
  | "disabled"
  | "ariaLabel"
  | "fullWidth"
  | "className"
  | "iconOnly";

type ButtonPassthroughProps = Omit<ButtonAsButton, SharedButtonPropKeys>;
type LinkPassthroughProps = Omit<ButtonAsLink, SharedButtonPropKeys>;

export const Button = forwardRef<HTMLButtonElement | HTMLAnchorElement, ButtonProps>(function Button(props, ref) {
  const {
    children,
    size = "sm",
    kind = "primary",
    icon,
    iconPosition = "left",
    loading = false,
    disabled = false,
    ariaLabel,
    fullWidth = false,
    className = "",
    iconOnly = false,
    ...rest
  } = props;
  const isIconOnly = Boolean(iconOnly || (!children && icon));
  const isLinkInteractive = !(disabled || loading);
  const sizeStyles = isIconOnly ? ICON_ONLY_SIZE_STYLES[size] : SIZE_STYLES[size];
  const kindStyles = KIND_STYLES[kind];
  const stateStyles = disabled ? DISABLED_STYLES : loading ? LOADING_STYLES : "";
  const widthStyles = fullWidth ? FULL_WIDTH_STYLES : "";

  const combinedClassName = twMerge(BASE_STYLES, sizeStyles, kindStyles, stateStyles, widthStyles, className);

  const spinnerSize = SPINNER_SIZE_MAP[size];
  const effectiveAriaLabel = ariaLabel || undefined;

  const content = (
    <>
      {loading && <Spinner size={spinnerSize} />}
      {isIconOnly && !loading && getButtonIcon(icon, size)}
      {!isIconOnly && (
        <>
          {!loading && iconPosition === "left" && getButtonIcon(icon, size)}
          {children && <span className="truncate">{children}</span>}
          {!loading && iconPosition === "right" && getButtonIcon(icon, size)}
        </>
      )}
    </>
  );

  const commonProps = getButtonCommonProps({
    className: combinedClassName,
    ariaLabel: effectiveAriaLabel,
    disabled,
    loading,
    size,
    kind,
    isIconOnly,
  });

  if (props.href !== undefined) {
    const { href, target, rel, onClick, ...restLinkProps } = rest as LinkPassthroughProps;
    const linkRel = target === "_blank" && !rel ? "noopener noreferrer" : rel;
    const linkProps = isLinkInteractive ? { href, target, rel: linkRel, onClick } : { tabIndex: -1 };
    return (
      <a ref={ref as React.Ref<HTMLAnchorElement>} {...linkProps} {...commonProps} {...restLinkProps}>
        {content}
      </a>
    );
  }

  const { type = "button", onClick, ...restButtonProps } = rest as ButtonPassthroughProps;

  return (
    <button
      ref={ref as React.Ref<HTMLButtonElement>}
      type={type}
      disabled={disabled || loading}
      onClick={disabled || loading ? undefined : onClick}
      {...commonProps}
      {...restButtonProps}
    >
      {content}
    </button>
  );
});
