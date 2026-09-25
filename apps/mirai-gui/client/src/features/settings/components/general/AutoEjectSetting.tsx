import { Toggle } from "@/components/ui";
import { Input as HeadlessInput } from "@headlessui/react";
import { useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";
import { SettingRow } from "./SettingRow";

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
      control={<Toggle checked={autoEjectEnabled} onChange={() => setAutoEjectEnabled(!autoEjectEnabled)} />}
    >
      <div className="mt-3 flex items-center gap-3">
        <label className="text-[13px] text-label-title dark:text-label-title-dark min-w-[140px]">
          Idle timeout (minutes)
        </label>
        <HeadlessInput
          type="text"
          inputMode="numeric"
          value={minutesInput}
          className={twMerge(
            "max-w-[72px] px-2 py-1 text-[13px] border border-cell-border dark:border-cell-border-dark bg-bg dark:bg-bg-dark placeholder:text-label-muted dark:placeholder:text-label-muted-dark rounded-[8px] focus:outline-none",
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
