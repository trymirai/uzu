import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Button as HeadlessButton } from "@headlessui/react";
import { twMerge } from "tailwind-merge";

export type IconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  icon?: ReactNode;
  children?: ReactNode;
};

export function IconButton({ icon, children, className, ...rest }: IconButtonProps) {
  return (
    <HeadlessButton
      {...rest}
      className={twMerge(
        "inline-flex items-center gap-1 rounded-md p-0.5 bg-transparent text-gray-1000 outline-hidden data-[focus]:shadow-focus",
        "hover:text-gray-1200 active:scale-[0.97] disabled:opacity-50 transition-colors duration-200",
        className,
      )}
    >
      {icon && <span className="grid place-items-center">{icon}</span>}
      {children}
    </HeadlessButton>
  );
}
