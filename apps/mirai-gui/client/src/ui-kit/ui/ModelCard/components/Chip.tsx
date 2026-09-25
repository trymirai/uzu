import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { Text } from "../../Typography";

export function Chip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={twMerge(
        "group/chip flex items-center justify-center h-7 px-2 py-0.5 rounded border-[0.5px] border-border-outlined bg-transparent hover:border-border-outlined-hover",
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
      <Text
        as="span"
        color="muted"
        opticalSize={14}
        className="text-[13px] font-[450] leading-[1.3] group-hover/chip:text-text-primary"
      >
        {children}
      </Text>
    </Chip>
  );
}
