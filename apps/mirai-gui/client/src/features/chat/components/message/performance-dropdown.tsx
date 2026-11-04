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
    <div className="flex flex-col gap-1 px-3 py-2">
      <div className="text-[15px] font-[500] leading-[1.2] text-label-title tabular-nums">{value}</div>
      <div className="text-[12px] leading-[1.3] text-label-muted whitespace-nowrap">{label}</div>
    </div>
  );
}

export function PerformanceDropdown({ perf, disabled = false }: PerformanceDropdownProps) {
  const tokensPerForwardPass = perf?.tokensPerForwardPass;
  const hasSpeculation = typeof tokensPerForwardPass === "number" && Number.isFinite(tokensPerForwardPass);
  return (
    <Popover className="w-fit">
      <PopoverButton
        disabled={disabled}
        aria-label="Performance"
        className="flex size-8 items-center justify-center rounded-lg text-label-muted transition-colors enabled:hover:bg-bg-hover enabled:hover:text-label-title focus:outline-hidden data-[focus]:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed"
      >
        <Activity className="size-4" aria-hidden="true" />
      </PopoverButton>
      <Transition
        enter="transition-[opacity,scale] duration-200 ease-spring motion-reduce:transition-none"
        enterFrom="scale-95 opacity-0"
        enterTo="scale-100 opacity-100"
        leave="transition-[opacity,scale] duration-[140ms] ease-spring motion-reduce:transition-none"
        leaveFrom="scale-100 opacity-100"
        leaveTo="scale-95 opacity-0"
      >
        <PopoverPanel
          anchor="top start"
          className="[--anchor-gap:8px] origin-bottom-left rounded-xl bg-surface-popover shadow-popover z-50 pointer-events-auto focus:outline-hidden focus:ring-0"
        >
          <div className="p-1">
            <div className={`grid ${hasSpeculation ? "grid-cols-4" : "grid-cols-3"}`}>
              <PerfStat value={formatSeconds(perf?.ttftSec)} label="Time to first token" />
              <PerfStat value={formatTps(perf?.tps)} label="Tokens per second" />
              {hasSpeculation && (
                <PerfStat value={`${tokensPerForwardPass.toFixed(2)} tok/pass`} label="Acceptance rate" />
              )}
              <PerfStat value={formatSeconds(perf?.totalSec)} label="Total time" />
            </div>
          </div>
        </PopoverPanel>
      </Transition>
    </Popover>
  );
}
