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
  /** A square segment holding just an icon. */
  icon?: boolean;
};

// The same filled pill track as the search field and the segmented control beside it; segments are
// pills on it that only show a fill under the pointer.
const Base: React.FC<GroupButtonProps> = ({ className, children }) => {
  return (
    <div className={twMerge("inline-flex h-8 items-center gap-0.5 rounded-full bg-control-fill p-0.5", className)}>
      {children}
    </div>
  );
};

const Segment: React.FC<SegmentProps> = ({ className, onClick, disabled = false, children, ariaLabel, icon }) => {
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
        "flex h-7 items-center gap-2 rounded-full text-[13px] font-[450] text-text-muted outline-hidden select-none",
        "transition-[background-color,color] duration-150 ease-out hover:bg-control-fill-hover hover:text-text-primary focus-visible:shadow-focus",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-text-muted",
        icon ? "w-7 justify-center" : "pl-2 pr-3",
        className,
      )}
    >
      {children}
    </button>
  );
};

export const GroupButton = Object.assign(Base, { Segment });
