import { Download, LoaderCircle, RefreshCw, X } from "lucide-react";
import { IconAction } from "@/components/ui/icon-action";
import { IconPauseFilled } from "@/components/icons/pause-filled-icon";
import { IconPlayFilled } from "@/components/icons/play-filled-icon";
import { Tooltip } from "@/components/ui/tooltip";
import type { ModelCardStatus } from "./types";

const ACTION_BUTTON_CLASS =
  "h-7 px-1.5 py-0.5 rounded bg-border-default text-text-primary hover:bg-control-surface-hover active:bg-control-surface-active";

type DownloadActionsProps = {
  status: ModelCardStatus;
  isDownloading: boolean;
  isPaused: boolean;
  isError: boolean;
  onDownload?: () => void;
  onPause?: () => void;
  onCancel?: () => void;
  onRetry?: () => void;
};

export function DownloadActions({
  status,
  isDownloading,
  isPaused,
  isError,
  onDownload,
  onPause,
  onCancel,
  onRetry,
}: DownloadActionsProps) {
  return (
    <>
      {status === "initializing" && (
        <span
          role="status"
          aria-label="Checking model"
          title="Checking model"
          className="flex h-7 w-7 items-center justify-center text-text-muted"
        >
          <LoaderCircle size={16} className="animate-spin motion-reduce:animate-none" />
        </span>
      )}
      {isError && onRetry && (
        <Tooltip content="Retry" side="top">
          <span>
            <IconAction
              label="Retry download"
              icon={<RefreshCw size={14} />}
              onClick={onRetry}
              className={ACTION_BUTTON_CLASS}
            />
          </span>
        </Tooltip>
      )}
      {status === "available" && onDownload && (
        <Tooltip content="Download" side="top">
          <span>
            <IconAction
              label="Download model"
              icon={<Download size={16} />}
              onClick={onDownload}
              className={ACTION_BUTTON_CLASS}
            />
          </span>
        </Tooltip>
      )}
      {isDownloading && (
        <>
          <Tooltip content={isPaused ? "Resume" : "Pause"} side="top">
            <span>
              <IconAction
                label={isPaused ? "Resume download" : "Pause download"}
                icon={isPaused ? <IconPlayFilled /> : <IconPauseFilled />}
                onClick={onPause}
                className={ACTION_BUTTON_CLASS}
              />
            </span>
          </Tooltip>
          {onCancel && (
            <Tooltip content="Cancel" side="top">
              <span>
                <IconAction
                  label="Cancel download"
                  icon={<X size={14} />}
                  onClick={onCancel}
                  className={ACTION_BUTTON_CLASS}
                />
              </span>
            </Tooltip>
          )}
        </>
      )}
    </>
  );
}
