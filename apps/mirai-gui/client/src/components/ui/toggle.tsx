import React from "react";
import { twMerge } from "tailwind-merge";

type ToggleProps = {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
};

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, disabled = false, className }) => {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={twMerge(
        "relative inline-flex min-h-6 min-w-11 items-center rounded-full transition-colors focus:outline-hidden focus:ring-0 focus:ring-offset-0",
        checked ? "bg-label-title dark:bg-label-title-dark" : "bg-button-border dark:bg-button-border-dark",
        disabled && "opacity-50 cursor-not-allowed",
        !disabled && "cursor-pointer",
        className,
      )}
    >
      <span
        className={twMerge(
          "inline-block h-5 w-5 px-0.5 transform rounded-full bg-white dark:bg-label-title transition-transform",
          checked ? "translate-x-[22px]" : "translate-x-[2px]",
        )}
      />
    </button>
  );
};
