import { Download, RefreshCw, X } from "lucide-react";
import { IconAction } from "../../CardPrimitives";
import { IconPauseFilled } from "../../IconPauseFilled";
import { IconPlayFilled } from "../../IconPlayFilled";
import { Tooltip } from "../../Tooltip";
import type { ModelCardStatus } from "../types";

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
