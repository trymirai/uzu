import React from "react";
import { twMerge } from "tailwind-merge";

type ToggleProps = {
  /** Accessible name; the switch renders no text of its own. */
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
};

export const Toggle: React.FC<ToggleProps> = ({ label, checked, onChange, disabled = false, className }) => {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={twMerge(
        "relative inline-flex min-h-6 min-w-11 items-center rounded-full transition-colors outline-hidden focus-visible:shadow-focus",
        checked ? "bg-label-title" : "bg-button-border",
        disabled && "opacity-50 cursor-not-allowed",
        !disabled && "cursor-pointer",
        className,
      )}
    >
      <span
        className={twMerge(
          "inline-block h-5 w-5 px-0.5 transform rounded-full bg-white dark:bg-black transition-transform",
          checked ? "translate-x-[22px]" : "translate-x-[2px]",
        )}
      />
    </button>
  );
};
