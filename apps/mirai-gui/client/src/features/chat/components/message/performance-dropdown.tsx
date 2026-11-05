import { IconGauge } from "@/components/icons/app-icons";
import { Fragment } from "react";
import { Tooltip } from "@/components/ui/tooltip";
import type { PerfStats } from "@/types/message";

type PerformanceDropdownProps = {
  perf?: PerfStats;
  disabled?: boolean;
};

const formatSeconds = (seconds?: number): string =>
  typeof seconds === "number" && Number.isFinite(seconds) ? `${seconds.toFixed(3)}s` : "—";

const formatTps = (tps?: number): string =>
  typeof tps === "number" && Number.isFinite(tps) ? `${Math.round(tps)}` : "—";

type Metric = { value: string; short: string; full: string };

/**
 * The run's speed figures in our tooltip, opened at once on hover and lined up under the button like
 * the Regenerate menu beside it. One figure per line, everything aligned left: the number, then its
 * short name; the full name stays on the abbreviation for screen readers and the title.
 */
export function PerformanceDropdown({ perf, disabled = false }: PerformanceDropdownProps) {
  const tokensPerForwardPass = perf?.tokensPerForwardPass;
  const metrics: Metric[] = [
    { value: formatSeconds(perf?.ttftSec), short: "TTFT", full: "Time to first token" },
    { value: formatTps(perf?.tps), short: "TPS", full: "Tokens per second" },
    ...(typeof tokensPerForwardPass === "number" && Number.isFinite(tokensPerForwardPass)
      ? [{ value: `${tokensPerForwardPass.toFixed(2)} tok/pass`, short: "MAL", full: "Mean accepted length" }]
      : []),
    { value: formatSeconds(perf?.totalSec), short: "Time", full: "Total time" },
  ];

  const button = (
    <button
      type="button"
      disabled={disabled}
      aria-label="Performance"
      className="flex size-8 items-center justify-center rounded-lg text-label-muted transition-colors duration-150 ease-out enabled:hover:bg-bg-hover enabled:hover:text-label-title outline-hidden focus-visible:shadow-focus disabled:opacity-60 disabled:cursor-not-allowed"
    >
      <IconGauge size={16} aria-hidden />
    </button>
  );
  if (disabled) return button;
  return (
    <Tooltip
      side="bottom"
      align="start"
      variant="panel"
      delay={0}
      content={
        <div className="grid grid-cols-[auto_auto] items-baseline gap-x-3 gap-y-2 text-[13px] leading-[18px]">
          {metrics.map((metric) => (
            <Fragment key={metric.short}>
              <span className="font-[500] text-text-primary tabular-nums">{metric.value}</span>
              <span className="font-[400] text-text-muted">
                {metric.short === metric.full ? (
                  metric.short
                ) : (
                  <abbr title={metric.full} className="no-underline">
                    {metric.short}
                  </abbr>
                )}
              </span>
            </Fragment>
          ))}
        </div>
      }
    >
      {button}
    </Tooltip>
  );
}
