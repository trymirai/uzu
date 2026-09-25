import type { ButtonHTMLAttributes, ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import type { IconButtonVariant } from "../utils/button";
import { ICON_BUTTON_VARIANT_MAP } from "../utils/button";

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  icon?: ReactNode;
  variant?: IconButtonVariant;
  children?: ReactNode;
};

export function IconButton({ icon, children, variant = "secondary", className, ...rest }: IconButtonProps) {
  const sizeCls = variant === "pill" ? "p-0.5" : "size-8 text-sm";
  const base = twMerge(
    "inline-flex items-center gap-1 rounded-md focus:outline-none disabled:opacity-50 transition-colors duration-200",
    sizeCls,
  );
  const mergedClassName = twMerge(base, ICON_BUTTON_VARIANT_MAP[variant], className);
  return (
    <button {...rest} className={mergedClassName}>
      {icon && <span className="grid place-items-center">{icon}</span>}
      {children}
    </button>
  );
}
