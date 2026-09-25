import React from "react";
import { twMerge } from "tailwind-merge";

type GroupButtonProps = {
  className?: string;
  children: React.ReactNode;
};

type SegmentProps = {
  className?: string;
  onClick?: (e?: React.MouseEvent) => void;
  disabled?: boolean;
  children: React.ReactNode;
  ariaLabel?: string;
  leftDivider?: boolean;
};

const Base: React.FC<GroupButtonProps> = ({ className, children }) => {
  return (
    <div
      className={twMerge(
        "h-8 flex items-center rounded-lg overflow-hidden border border-cell-border dark:border-cell-border-dark bg-transparent",
        className,
      )}
    >
      {children}
    </div>
  );
};

const Segment: React.FC<SegmentProps> = ({
  className,
  onClick,
  disabled = false,
  children,
  ariaLabel,
  leftDivider = false,
}) => {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      onClick={(e) => {
        if (disabled) return;
        onClick?.(e);
      }}
      disabled={disabled}
      className={twMerge(
        "px-2 h-8 flex items-center gap-2 transition-colors select-none",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        "hover:bg-card-modal-hover dark:hover:bg-card-modal-hover-dark",
        leftDivider && "border-l-[0.5px] border-button-border dark:border-button-border-dark",
        className,
      )}
    >
      {children}
    </button>
  );
};

export const GroupButton = Object.assign(Base, { Segment });
