import { Popover, PopoverButton, PopoverPanel } from "@headlessui/react";
import { Toggle } from "@/components/ui/toggle";
import { SettingRow } from "./setting-row";

export function AnalyticsSetting({ enabled, onChange }: { enabled: boolean; onChange: (enabled: boolean) => void }) {
  return (
    <SettingRow
      title="Share usage analytics"
      description={
        <>
          Send model activity, performance measurements, OS/CPU/RAM, app versions, and a temporary session ID to Mirai.{" "}
          <Popover as="span">
            <PopoverButton
              type="button"
              className="rounded-sm underline underline-offset-2 decoration-label-muted/50 hover:text-label-title transition-colors outline-hidden data-[focus]:shadow-focus"
            >
              See more
            </PopoverButton>
            <PopoverPanel
              anchor="bottom end"
              focus
              transition
              role="region"
              aria-label="Usage analytics details"
              className="z-50 w-[min(28rem,calc(100vw-2rem))] rounded-lg bg-surface-elevated shadow-sm text-[13px] leading-[150%] text-label-muted outline-hidden [--anchor-gap:8px] [--anchor-padding:16px] [--anchor-max-height:min(70vh,36rem)] origin-top-right transition-[opacity,transform] duration-[120ms] ease-spring data-[closed]:opacity-0 data-[closed]:scale-95"
            >
              <div
                tabIndex={0}
                className="max-h-[inherit] overflow-y-auto overscroll-y-contain thin-scrollbar rounded-lg p-4 outline-hidden"
              >
                <ul className="list-disc space-y-2 pl-4">
                  <li>OS name and version, CPU model, total RAM, and app, uzu engine, and model toolchain versions.</li>
                  <li>
                    Model identifiers, download and generation start/finish events, UTC timestamps, and a generic
                    generation-failed event.
                  </li>
                  <li>
                    Reply duration, time to first token, input-processing and generation speeds,
                    input/cached-input/output token counts, memory used, tokens per decoding pass, and decoding pass
                    count, when available.
                  </li>
                  <li>
                    Input/output energy usage and joules per token, including CPU, GPU, Neural Engine, and RAM
                    breakdowns when available.
                  </li>
                  <li>A random session ID, renewed on app restart or when analytics is re-enabled.</li>
                </ul>
                <p className="mt-3">
                  Chat content, attachments, chat names, local paths, and detailed error messages are not sent. Mirai’s
                  server also sees your connection’s IP address.
                </p>
              </div>
            </PopoverPanel>
          </Popover>
        </>
      }
      control={<Toggle label="Share usage analytics" checked={enabled} onChange={onChange} />}
    />
  );
}
