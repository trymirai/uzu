import type { ComponentPropsWithoutRef } from "react";
import { twMerge } from "tailwind-merge";

type SelectionPanelProps = ComponentPropsWithoutRef<"div">;

export function SelectionPanel({ className, ...props }: SelectionPanelProps) {
  return (
    <div
      className={twMerge("rounded-lg bg-surface-elevated shadow-sm outline-hidden origin-bottom-right", className)}
      {...props}
    />
  );
}
