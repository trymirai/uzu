import type { ReactNode, HTMLAttributes } from "react";
import { twMerge } from "tailwind-merge";

type CardContainerProps = {
  children: ReactNode;
} & HTMLAttributes<HTMLDivElement>;

export function CardContainer({ children, className, ...props }: CardContainerProps) {
  return (
    <div className={twMerge("bg-bg-modal border border-cell-border rounded-lg overflow-clip", className)} {...props}>
      {children}
    </div>
  );
}
