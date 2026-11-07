import React from "react";
import { twMerge } from "tailwind-merge";
import { IconArrowDownCircle, IconCrossSmall } from "@/components/icons/app-icons";
import { Button } from "@/components/ui/button";
import {
  SURFACE_BUTTON_SMALL_CLASSNAME,
  SURFACE_CLOSE_CLASSNAME,
  SURFACE_SOFT_BUTTON_CLASSNAME,
} from "@/components/ui/modal/constants";

export type UpdateToastProps = {
  version: string;
  onApplyNow: () => void;
  onLater: () => void;
  errorMessage?: string;
  onClose?: () => void;
  visible?: boolean;
};

/**
 * The update notice in the corner: the same lifted surface and motion as the other toasts, with the
 * two choices as pills under the text.
 */
export const UpdateToast: React.FC<UpdateToastProps> = ({
  version,
  onApplyNow,
  onLater,
  errorMessage,
  onClose,
  visible = true,
}) => {
  return (
    <div
      role="status"
      className={twMerge(
        "pointer-events-auto flex w-[360px] max-w-[calc(100vw-2rem)] gap-3 rounded-[12px] bg-surface-popover p-4 pr-3 shadow-popover",
        visible ? "animate-uzu-toast-in" : "animate-uzu-toast-out",
      )}
    >
      <IconArrowDownCircle size={18} aria-hidden className="mt-px shrink-0 text-text-secondary" />

      <div className="flex min-w-0 flex-1 flex-col">
        <p className="text-[14px] font-[500] leading-[1.35] text-text-primary tabular-nums">
          Update {version} is ready
        </p>
        <p className="mt-0.5 text-[13px] font-[400] leading-[1.45] text-text-secondary text-pretty">
          Restart to finish installing it.
        </p>
        {errorMessage ? (
          <p className="mt-1.5 text-[13px] font-[400] leading-[1.45] text-danger text-pretty">{errorMessage}</p>
        ) : null}

        <div className="mt-3 flex items-center gap-2">
          <Button size="sm" kind="primary" onClick={onApplyNow} className={SURFACE_BUTTON_SMALL_CLASSNAME}>
            Restart now
          </Button>
          <Button
            size="sm"
            kind="secondary"
            onClick={onLater}
            className={twMerge(SURFACE_BUTTON_SMALL_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME)}
          >
            Later
          </Button>
        </div>
      </div>

      <button
        type="button"
        onClick={onClose}
        aria-label="Dismiss"
        className={twMerge(SURFACE_CLOSE_CLASSNAME, "-mt-1 size-6")}
      >
        <IconCrossSmall size={16} aria-hidden />
      </button>
    </div>
  );
};
