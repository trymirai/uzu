import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";

export type SegmentedControlOption = {
  value: string;
  label: ReactNode;
  disabled?: boolean;
  ariaLabel?: string;
};

export type SegmentedControlProps = {
  value: string;
  onChange: (value: string) => void;
  options: ReadonlyArray<SegmentedControlOption>;
  // Required: a radiogroup without an accessible name is announced as unlabelled.
  ariaLabel: string;
};

const CONTAINER_CLASSES =
  "inline-flex w-fit h-7 items-center rounded-lg overflow-hidden border border-border-default bg-transparent";

export function SegmentedControl({ value, onChange, options, ariaLabel }: SegmentedControlProps) {
  const handleSelect = (option: SegmentedControlOption) => {
    if (option.disabled || option.value === value) return;
    onChange(option.value);
  };

  return (
    <div role="radiogroup" aria-label={ariaLabel} className={CONTAINER_CLASSES}>
      {options.map((option, index) => {
        const isSelected = option.value === value;
        const segmentClasses = twMerge(
          "h-full px-2 inline-flex items-center gap-2 text-xs transition-colors select-none",
          index > 0 ? "border-l border-border-default" : undefined,
          isSelected ? "bg-surface-tertiary text-text-primary" : "text-text-muted hover:bg-surface-tertiary",
          option.disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
        );

        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={isSelected}
            aria-label={option.ariaLabel}
            disabled={option.disabled}
            onClick={() => handleSelect(option)}
            className={segmentClasses}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
