import { Toggle } from "@/components/ui/toggle";
import { Input as HeadlessInput } from "@headlessui/react";
import { useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";
import { SettingRow } from "./setting-row";

const MIN_IDLE_MINUTES = 1;
const MAX_IDLE_MINUTES = 240;

export function AutoEjectSetting({
  autoEjectEnabled,
  autoEjectMinutes,
  setAutoEjectEnabled,
  setAutoEjectMinutes,
}: {
  autoEjectEnabled: boolean;
  autoEjectMinutes: number;
  setAutoEjectEnabled: (value: boolean) => Promise<void>;
  setAutoEjectMinutes: (minutes: number) => Promise<void>;
}) {
  const [minutesInput, setMinutesInput] = useState(String(autoEjectMinutes));

  useEffect(() => {
    setMinutesInput(String(autoEjectMinutes));
  }, [autoEjectMinutes]);

  return (
    <SettingRow
      title="Auto‑eject models when idle"
      description="Automatically unload local models after inactivity"
      control={
        <Toggle label="Auto-eject" checked={autoEjectEnabled} onChange={() => setAutoEjectEnabled(!autoEjectEnabled)} />
      }
    >
      <div className="mt-3 flex items-center gap-3">
        <label className="text-[13px] text-label-title min-w-[140px]">Idle timeout (minutes)</label>
        <HeadlessInput
          type="text"
          inputMode="numeric"
          value={minutesInput}
          className={twMerge(
            "h-8 w-[72px] px-3 text-[13px] tabular-nums text-text-primary rounded-lg border-[0.5px] border-transparent bg-control-fill placeholder:text-label-muted outline-hidden transition-colors duration-150 ease-out focus:border-border-outlined",
            !autoEjectEnabled ? "opacity-60 cursor-not-allowed" : "",
          )}
          onChange={(e) => {
            setMinutesInput(e.target.value);
          }}
          onBlur={async () => {
            const parsed = Number(minutesInput);
            const clamped = Number.isFinite(parsed)
              ? Math.max(MIN_IDLE_MINUTES, Math.min(MAX_IDLE_MINUTES, Math.round(parsed)))
              : autoEjectMinutes;
            setMinutesInput(String(clamped));
            await setAutoEjectMinutes(clamped);
          }}
          disabled={!autoEjectEnabled}
          aria-disabled={!autoEjectEnabled}
        />
      </div>
    </SettingRow>
  );
}
