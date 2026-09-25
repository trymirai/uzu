import React from "react";
import { twMerge } from "tailwind-merge";
import { Button } from "@/ui-kit";
import { X } from "lucide-react";

export type UpdateToastProps = {
  version: string;
  onApplyNow: () => void;
  onLater: () => void;
  errorMessage?: string;
  onClose?: () => void;
};

export const UpdateToast: React.FC<UpdateToastProps> = ({ version, onApplyNow, onLater, errorMessage, onClose }) => {
  return (
    <div
      className={twMerge(
        "w-[560px] max-w-[90vw] bg-card-modal dark:bg-card-modal-dark border border-button-border dark:border-button-border-dark rounded-xl p-4 shadow-lg",
        "flex flex-col gap-3",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col">
          <div className="text-label-title dark:text-label-title-dark font-semibold">Update {version} is ready</div>
          <div className="text-label-muted dark:text-label-muted-dark text-sm mt-1">Restart to apply the update.</div>
          {errorMessage ? <div className="text-sm mt-2 text-error">{errorMessage}</div> : null}
        </div>
        <button
          onClick={onClose}
          className="text-label-muted dark:text-label-muted-dark hover:text-label-title dark:hover:text-label-title-dark p-1 rounded"
          aria-label="Close"
        >
          <X size={18} />
        </button>
      </div>

      <div className="flex items-center gap-2 mt-1">
        <Button size="sm" kind="primary" onClick={onApplyNow}>
          Restart now
        </Button>
        <Button size="sm" kind="secondary" onClick={onLater}>
          Later
        </Button>
      </div>
    </div>
  );
};
