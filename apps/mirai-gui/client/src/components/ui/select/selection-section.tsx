import type { ComponentPropsWithoutRef } from "react";
import { twMerge } from "tailwind-merge";

type SelectionSectionProps = ComponentPropsWithoutRef<"div"> & {
  divided?: boolean;
};

export function SelectionSection({ divided = false, className, ...props }: SelectionSectionProps) {
  return (
    <div
      className={twMerge(
        "px-1 flex flex-col gap-px",
        divided
          ? "mt-1 pt-1 relative before:absolute before:inset-x-3 before:top-0 before:h-px before:bg-border-default"
          : undefined,
        className,
      )}
      {...props}
    />
  );
}
