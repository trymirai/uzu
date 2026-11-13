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

// A pill track; the chosen segment sits raised on it, like a key pressed into place.
const CONTAINER_CLASSES = "inline-flex w-fit h-8 items-center gap-0.5 rounded-full bg-control-fill p-0.5";

export function SegmentedControl({ value, onChange, options, ariaLabel }: SegmentedControlProps) {
  const handleSelect = (option: SegmentedControlOption) => {
    if (option.disabled || option.value === value) return;
    onChange(option.value);
  };

  return (
    <div role="radiogroup" aria-label={ariaLabel} className={CONTAINER_CLASSES}>
      {options.map((option) => {
        const isSelected = option.value === value;
        const segmentClasses = twMerge(
          "h-full px-3 inline-flex items-center gap-2 rounded-full text-[13px] font-[450] transition-[background-color,color,box-shadow] duration-150 ease-out select-none outline-hidden focus-visible:shadow-focus",
          isSelected
            ? "bg-control-selected text-text-primary shadow-[0_1px_2px_rgb(0_0_0/0.08),0_0_0_0.5px_rgb(0_0_0/0.06)] dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)]"
            : "text-text-muted hover:text-text-primary",
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
