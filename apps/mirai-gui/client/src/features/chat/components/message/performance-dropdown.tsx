import { Popover, PopoverButton, PopoverPanel, Transition } from "@headlessui/react";
import { twMerge } from "tailwind-merge";
import { PerformanceIcon } from "@/components/icons/performance-icon";
import type { PerfStats } from "@/types/message";

type PerformanceDropdownProps = {
  perf?: PerfStats;
  disabled?: boolean;
};

const formatSeconds = (seconds?: number): string =>
  typeof seconds === "number" && Number.isFinite(seconds) ? `${seconds.toFixed(3)}s` : "—";

const formatTps = (tps?: number): string =>
  typeof tps === "number" && Number.isFinite(tps) ? `${Math.round(tps)}` : "—";

const BUTTON_BASE =
  "flex items-center px-[6px] transition-colors focus:outline-none focus:ring-0 gap-2 text-label-muted dark:text-label-muted-dark rounded-md py-1";
const BUTTON_ENABLED =
  "group-hover:text-label-title dark:group-hover:text-label-title-dark group-hover:bg-bg-hover dark:group-hover:bg-bg-hover-dark";
const BUTTON_DISABLED = "opacity-60 cursor-not-allowed";

function PerfStat({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="text-[15px] leading-[120%] text-label-title dark:text-label-title-dark">{value}</div>
      <div className="text-[11px] leading-[120%] text-label-muted dark:text-label-muted-dark mt-1">{label}</div>
    </div>
  );
}

export function PerformanceDropdown({ perf, disabled = false }: PerformanceDropdownProps) {
  return (
    <Popover className="group w-fit">
      <PopoverButton disabled={disabled} className={twMerge(BUTTON_BASE, disabled ? BUTTON_DISABLED : BUTTON_ENABLED)}>
        <PerformanceIcon />
        <span className="text-[13px]">Performance</span>
      </PopoverButton>
      <Transition
        enter="transition duration-100 ease-out"
        enterFrom="transform scale-95 opacity-0"
        enterTo="transform scale-100 opacity-100"
        leave="transition duration-75 ease-out"
        leaveFrom="transform scale-100 opacity-100"
        leaveTo="transform scale-95 opacity-0"
      >
        <PopoverPanel
          anchor="bottom end"
          className="[--anchor-gap:8px] bg-bg-modal dark:bg-bg-modal-dark border border-cell-border dark:border-cell-border-dark rounded-[6px] z-50 pointer-events-auto focus:outline-none focus:ring-0"
        >
          <div className="p-[10px]">
            <div className="grid grid-cols-3 gap-3 text-center">
              <PerfStat value={formatSeconds(perf?.ttftSec)} label="Time to first token" />
              <PerfStat value={formatTps(perf?.tps)} label="Tokens per second" />
              <PerfStat value={formatSeconds(perf?.totalSec)} label="Total time" />
            </div>
          </div>
        </PopoverPanel>
      </Transition>
    </Popover>
  );
}
