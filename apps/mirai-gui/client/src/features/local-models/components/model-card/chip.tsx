import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { Text } from "@/components/ui/typography";

function Chip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={twMerge(
        "group/chip flex items-center justify-center h-6 px-2.5 rounded-full bg-surface-popover shadow-[0_0_0_0.5px_rgb(0_0_0/0.08),0_1px_2px_rgb(0_0_0/0.06)] dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function ChipText({ children }: { children: ReactNode }) {
  return (
    <Chip>
      <Text as="span" color="muted" opticalSize={14} className="text-[12px] font-[450] leading-[1.3] tabular-nums">
        {children}
      </Text>
    </Chip>
  );
}
