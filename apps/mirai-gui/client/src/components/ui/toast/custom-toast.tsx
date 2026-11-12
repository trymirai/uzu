import { CircleCheckIcon, InfoIcon, XIcon, CircleAlertIcon } from "@/components/icons/app-icons";
import type { ComponentType, ReactNode } from "react";
import type { Toast } from "react-hot-toast";
import toast from "react-hot-toast";
import { twMerge } from "tailwind-merge";
import type { ToastType } from "./types";

export type CustomToastProps = {
  t: Toast;
  message: ReactNode;
  description?: string;
  type: ToastType;
  onClick?: () => void;
};

const TOAST_ICON_BY_TYPE: Record<
  ToastType,
  { Icon: ComponentType<{ size?: number; className?: string }>; tone: string }
> = {
  success: { Icon: CircleCheckIcon, tone: "text-success" },
  info: { Icon: InfoIcon, tone: "text-text-muted" },
  warning: { Icon: CircleAlertIcon, tone: "text-amber-500" },
  error: { Icon: CircleAlertIcon, tone: "text-danger" },
};

/**
 * A note in the corner: the same floating surface as menus, one small coloured icon for the kind of
 * message, and the text, with an optional quieter line under it. The icon and the close button line
 * up with the first line, however many lines follow. It comes into focus from just above and leaves a
 * little faster than it came.
 */
export function CustomToast({ t, message, description, type, onClick }: CustomToastProps) {
  const isClickable = onClick !== undefined;
  const { Icon, tone } = TOAST_ICON_BY_TYPE[type];

  return (
    <div
      className={twMerge(
        // The same card as the update notice beside it: 12px corners on the floating surface.
        "pointer-events-auto flex min-h-11 max-w-[min(360px,calc(100vw-2rem))] items-start gap-2.5 rounded-[12px] bg-surface-popover py-2 pl-3.5 pr-2 shadow-popover",
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
      <span className="flex h-7 shrink-0 items-center">
        <Icon size={16} className={tone} aria-hidden />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-0.5 py-1.5">
        <span className="text-[13px] font-[450] leading-4 text-text-primary text-pretty">{message}</span>
        {description ? (
          <span className="line-clamp-3 text-[12px] font-[400] leading-4 text-text-muted text-pretty">
            {description}
          </span>
        ) : null}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          toast.dismiss(t.id);
        }}
        aria-label="Dismiss"
        className="relative mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md text-text-muted outline-hidden transition-colors duration-150 ease-out hover:bg-selection-hover hover:text-text-primary focus-visible:shadow-focus before:absolute before:-inset-2"
      >
        <XIcon size={16} aria-hidden />
      </button>
    </div>
  );
}
