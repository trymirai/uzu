import type { ComponentPropsWithoutRef } from "react";
import { twMerge } from "tailwind-merge";

type SelectionPanelProps = ComponentPropsWithoutRef<"div">;

export function SelectionPanel({ className, ...props }: SelectionPanelProps) {
  return (
    <div
      className={twMerge("rounded-xl bg-surface-popover shadow-popover outline-hidden origin-bottom-right", className)}
      {...props}
    />
  );
}
