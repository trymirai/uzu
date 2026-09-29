import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";

export type ButtonSize = "xxs" | "xs" | "sm" | "lg";
export type ButtonKind = "primary" | "secondary" | "danger" | "ghost";
export type IconPosition = "left" | "right";
export type LinkTarget = "_blank" | "_self" | "_parent" | "_top";

type ButtonSharedProps = {
  children?: ReactNode;
  size?: ButtonSize;
  kind?: ButtonKind;
  icon?: ReactNode;
  iconPosition?: IconPosition;
  loading?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  fullWidth?: boolean;
  className?: string;
  iconOnly?: boolean;
};

export type ButtonAsButton = ButtonSharedProps &
  Omit<
    ButtonHTMLAttributes<HTMLButtonElement>,
    "className" | "children" | "aria-label" | "aria-describedby" | "disabled" | "title"
  > & {
    href?: undefined;
  };

export type ButtonAsLink = ButtonSharedProps &
  Omit<
    AnchorHTMLAttributes<HTMLAnchorElement>,
    "className" | "children" | "aria-label" | "aria-describedby" | "title"
  > & {
    href: string;
    target?: LinkTarget;
    rel?: string;
  };

export type ButtonProps = ButtonAsButton | ButtonAsLink;
