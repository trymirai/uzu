import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { twMerge } from "tailwind-merge";

// The same square as the Copy, Regenerate and speed buttons beside it: hover fills it, focus rings it.
const ARROW_CLASSNAME =
  "flex size-8 items-center justify-center rounded-lg text-label-muted transition-colors duration-150 ease-out enabled:hover:bg-bg-hover enabled:hover:text-label-title outline-hidden focus-visible:shadow-focus disabled:opacity-40 disabled:cursor-not-allowed";

type MessageVersionControlsProps = {
  currentVersion: number;
  totalVersions: number;
  onVersionChange: (versionIndex: number) => void;
  className?: string;
  disabledPrevious?: boolean;
  currentModelName?: string;
};

export const MessageVersionControls: React.FC<MessageVersionControlsProps> = ({
  currentVersion,
  totalVersions,
  onVersionChange,
  className,
  disabledPrevious,
  currentModelName,
}) => {
  const canGoPrevious = currentVersion > 0 && !disabledPrevious;
  const canGoNext = currentVersion < totalVersions - 1;

  const handlePrevious = () => {
    if (canGoPrevious) {
      onVersionChange(currentVersion - 1);
    }
  };

  const handleNext = () => {
    if (canGoNext) {
      onVersionChange(currentVersion + 1);
    }
  };

  if (totalVersions <= 1) {
    return null;
  }

  return (
    <div className={twMerge("flex items-center", className)}>
      <button
        type="button"
        onClick={handlePrevious}
        disabled={!canGoPrevious}
        aria-label="Previous response"
        className={ARROW_CLASSNAME}
      >
        <ChevronLeft aria-hidden className="size-4" />
      </button>

      <span
        className="min-w-7 text-center text-[13px] font-[450] leading-[1.3] text-label-muted tabular-nums"
        title={currentModelName || undefined}
      >
        {currentVersion + 1}/{totalVersions}
      </span>

      <button
        type="button"
        onClick={handleNext}
        disabled={!canGoNext}
        aria-label="Next response"
        className={ARROW_CLASSNAME}
      >
        <ChevronRight aria-hidden className="size-4" />
      </button>
    </div>
  );
};
