import { twMerge } from "tailwind-merge";
import { IconCheckmark } from "../../icons/checkmark-icon";
import { CHECKBOX_SIZE, CHECKED_BG, CHECKED_BORDER, FOCUS_RING, UNCHECKED_BORDER } from "./constants";
import type { CheckboxProps } from "./types";

export function Checkbox({
  checked,
  onChange,
  size = "md",
  disabled = false,
  className,
  "aria-label": ariaLabel,
}: CheckboxProps) {
  const sizeConfig = CHECKBOX_SIZE[size];

  const boxClasses = twMerge(
    "relative m-0 block shrink-0 appearance-none border-[1.5px] p-0 outline-none transition-all duration-150 ease-out align-middle cursor-pointer",
    sizeConfig.box,
    sizeConfig.radius,
    checked ? `${CHECKED_BG} ${CHECKED_BORDER}` : UNCHECKED_BORDER,
    FOCUS_RING,
    className,
  );

  return (
    <span className="relative inline-flex shrink-0 items-center justify-center">
      <input
        type="checkbox"
        className={boxClasses}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        aria-checked={checked}
        aria-label={ariaLabel}
      />
      <IconCheckmark
        strokeWidth={2.5}
        size={sizeConfig.icon}
        className={twMerge(
          "pointer-events-none absolute transition-opacity duration-100 stroke-current",
          "text-primary-contrast",
          checked ? "opacity-100" : "opacity-0",
        )}
      />
    </span>
  );
}
