import type { ComponentPropsWithoutRef } from "react";
import { twMerge } from "tailwind-merge";

type SelectionSectionProps = ComponentPropsWithoutRef<"div"> & {
  divided?: boolean;
};

export function SelectionSection({ divided = false, className, ...props }: SelectionSectionProps) {
  return (
    <div
      className={twMerge(
        "px-1.5 flex flex-col gap-0.5",
        divided ? "border-t border-border-default mt-1" : undefined,
        className,
      )}
      {...props}
    />
  );
}
