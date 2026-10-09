import { Popover, PopoverButton, PopoverPanel, Transition } from "@headlessui/react";
import { Activity } from "lucide-react";
import type { PerfStats } from "@/types/message";

type PerformanceDropdownProps = {
  perf?: PerfStats;
  disabled?: boolean;
};

const formatSeconds = (seconds?: number): string =>
  typeof seconds === "number" && Number.isFinite(seconds) ? `${seconds.toFixed(3)}s` : "—";

const formatTps = (tps?: number): string =>
  typeof tps === "number" && Number.isFinite(tps) ? `${Math.round(tps)}` : "—";

function PerfStat({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="text-[15px] leading-[120%] text-label-title">{value}</div>
      <div className="text-[11px] leading-[120%] text-label-muted mt-1">{label}</div>
    </div>
  );
}

export function PerformanceDropdown({ perf, disabled = false }: PerformanceDropdownProps) {
  return (
    <Popover className="w-fit">
      <PopoverButton
        disabled={disabled}
        aria-label="Performance"
        title="Performance"
        className="flex size-8 items-center justify-center rounded-md text-label-muted transition-colors enabled:hover:bg-bg-hover enabled:hover:text-label-title focus:outline-hidden focus-visible:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed"
      >
        <Activity className="size-4" aria-hidden="true" />
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
          anchor="top start"
          className="[--anchor-gap:8px] bg-bg-modal border border-cell-border rounded-[6px] z-50 pointer-events-auto focus:outline-hidden focus:ring-0"
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
