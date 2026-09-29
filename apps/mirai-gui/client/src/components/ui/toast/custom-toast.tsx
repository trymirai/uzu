import { Info, TriangleAlert, X } from "lucide-react";
import type { ReactNode } from "react";
import type { Toast } from "react-hot-toast";
import toast from "react-hot-toast";
import { twMerge } from "tailwind-merge";
import { IconButton } from "../icon-button";
import { IconCheckCircleFilled } from "../../icons/check-circle-filled-icon";
import type { ToastType } from "./types";

export type CustomToastProps = {
  t: Toast;
  message: ReactNode;
  type: ToastType;
  onClick?: () => void;
};

type ToastIconConfig = {
  Icon: (props: { className?: string; size?: number }) => ReactNode;
  className: string;
};

const TOAST_ICON_BY_TYPE: Record<ToastType, ToastIconConfig> = {
  success: { Icon: IconCheckCircleFilled, className: "text-success" },
  info: { Icon: Info, className: "text-blue-500" },
  warning: { Icon: TriangleAlert, className: "text-yellow-500" },
  error: { Icon: TriangleAlert, className: "text-error" },
};

function ToastIcon({ type }: { type: ToastType }) {
  const { Icon, className } = TOAST_ICON_BY_TYPE[type];
  return <Icon className={className} size={20} />;
}

export function CustomToast({ t, message, type, onClick }: CustomToastProps) {
  const isClickable = onClick !== undefined;

  return (
    <div
      className={twMerge(
        "flex items-center justify-between rounded-lg border border-gray-500 bg-surface-elevated py-2 pl-3 pr-2 shadow-custom transition-opacity",
        t.visible ? "opacity-100" : "opacity-0",
        isClickable && "cursor-pointer hover:bg-surface-tertiary",
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
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
    >
      <div className="flex min-w-0 items-center">
        <ToastIcon type={type} />
        <span className="ml-2 mr-5 min-w-0 truncate text-sm font-medium text-text-primary">{message}</span>
      </div>
      <IconButton
        onClick={(e) => {
          e.stopPropagation();
          toast.dismiss(t.id);
        }}
        className="text-text-muted hover:text-text-primary"
        icon={<X size={20} />}
        aria-label="Dismiss toast"
      />
    </div>
  );
}
