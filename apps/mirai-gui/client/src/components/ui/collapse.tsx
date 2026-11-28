import { Transition } from "@headlessui/react";
import type { ReactNode } from "react";

export function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <Transition
      show={open}
      enter="transition-[grid-template-rows,opacity] duration-250 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none"
      enterFrom="grid-rows-[0fr] opacity-0"
      enterTo="grid-rows-[1fr] opacity-100"
      leave="transition-[grid-template-rows,opacity] duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none"
      leaveFrom="grid-rows-[1fr] opacity-100"
      leaveTo="grid-rows-[0fr] opacity-0"
    >
      <div className="grid">
        <div className="min-h-0 min-w-0 overflow-clip">{children}</div>
      </div>
    </Transition>
  );
}
