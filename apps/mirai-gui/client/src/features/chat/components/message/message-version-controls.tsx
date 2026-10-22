import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { twMerge } from "tailwind-merge";

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
    <div className={twMerge("flex items-center gap-1.5", className)}>
      <button
        onClick={handlePrevious}
        disabled={!canGoPrevious}
        aria-label="Previous response"
        title="Previous response"
        className={twMerge(
          "p-1 rounded transition-colors",
          canGoPrevious ? "text-label-muted hover:text-label-title" : "text-label-muted/50 cursor-not-allowed",
        )}
      >
        <ChevronLeft className="w-4 h-4" />
      </button>

      <span className="text-sm text-label-muted" title={currentModelName || undefined}>
        {currentVersion + 1}/{totalVersions}
      </span>

      <button
        onClick={handleNext}
        disabled={!canGoNext}
        aria-label="Next response"
        title="Next response"
        className={twMerge(
          "p-1 rounded transition-colors",
          canGoNext ? "text-label-muted hover:text-label-title" : "text-label-muted/50 cursor-not-allowed",
        )}
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  );
};
