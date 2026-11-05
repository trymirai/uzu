import {
  IconCircleCheck,
  IconCircleInfo,
  IconCrossSmall,
  IconExclamationCircle,
  IconExclamationTriangle,
} from "@/components/icons/app-icons";
import type { ComponentType, ReactNode } from "react";
import type { Toast } from "react-hot-toast";
import toast from "react-hot-toast";
import { twMerge } from "tailwind-merge";
import type { ToastType } from "./types";

export type CustomToastProps = {
  t: Toast;
  message: ReactNode;
  type: ToastType;
  onClick?: () => void;
};

const TOAST_ICON_BY_TYPE: Record<
  ToastType,
  { Icon: ComponentType<{ size?: number; className?: string }>; tone: string }
> = {
  success: { Icon: IconCircleCheck, tone: "text-success" },
  info: { Icon: IconCircleInfo, tone: "text-text-muted" },
  warning: { Icon: IconExclamationTriangle, tone: "text-amber-500" },
  error: { Icon: IconExclamationCircle, tone: "text-danger" },
};

/**
 * A note in the corner: the same floating surface as menus, one small coloured icon for the kind of
 * message, and the text. It comes into focus from just above and leaves a little faster than it came.
 */
export function CustomToast({ t, message, type, onClick }: CustomToastProps) {
  const isClickable = onClick !== undefined;
  const { Icon, tone } = TOAST_ICON_BY_TYPE[type];

  return (
    <div
      className={twMerge(
        // The same card as the update notice beside it: 12px corners on the floating surface.
        "pointer-events-auto flex min-h-11 max-w-[380px] items-center gap-2.5 rounded-[12px] bg-surface-popover py-2 pl-3.5 pr-2 shadow-popover",
        t.visible ? "animate-uzu-toast-in" : "animate-uzu-toast-out",
        isClickable && "cursor-pointer transition-colors duration-150 ease-out hover:bg-selection-hover",
      )}
      onClick={isClickable ? onClick : undefined}
      onKeyDown={
        isClickable
          ? (event) => {
              if (event.target !== event.currentTarget) return;
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
      role={isClickable ? "button" : "status"}
      tabIndex={isClickable ? 0 : undefined}
    >
      <Icon size={16} className={twMerge("shrink-0", tone)} aria-hidden />
      <span className="min-w-0 flex-1 text-[13px] font-[450] leading-[1.35] text-text-primary text-pretty">
        {message}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          toast.dismiss(t.id);
        }}
        aria-label="Dismiss"
        className="relative flex size-6 shrink-0 items-center justify-center rounded-md text-text-muted outline-hidden transition-colors duration-150 ease-out hover:bg-selection-hover hover:text-text-primary focus-visible:shadow-focus before:absolute before:-inset-2"
      >
        <IconCrossSmall size={16} aria-hidden />
      </button>
    </div>
  );
}
