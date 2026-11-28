import type { ReactNode } from "react";

/** Copying a message leaves these rows out. */
export function ToolRow({ children }: { children: ReactNode }) {
  return (
    <div
      data-tool-call
      className="mt-1 mb-2 flex h-6 min-w-0 items-center gap-2 pl-1 text-[13px] text-label-muted animate-uzu-in"
    >
      {children}
    </div>
  );
}
