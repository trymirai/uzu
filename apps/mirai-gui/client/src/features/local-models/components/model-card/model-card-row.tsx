import type { KeyboardEvent } from "react";
import { Button, DataInteractive } from "@headlessui/react";
import { Trash2 } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { IconCheckStroke } from "@/components/icons/check-stroke-icon";
import { Tooltip } from "@/components/ui/tooltip";
import { Text } from "@/components/ui/typography";
import type { ModelCardProps } from "./types";
import { deriveModelCardState } from "./model-card-state";
import { DownloadActions } from "./download-actions";
import { ModelBadges } from "./model-badges";
import { ModelHeading } from "./model-heading";

export function ModelCardRow({
  name,
  logo,
  parameters,
  size,
  state,
  onDownload,
  onPause,
  onCancel,
  onDelete,
  onRetry,
  onOpen,
  compact = false,
}: ModelCardProps) {
  const { status, isDownloading, isDownloaded, isPaused, isError, progress } = deriveModelCardState(state);

  return (
    <DataInteractive
      as="div"
      className={twMerge(
        "relative py-2.5 transition-colors duration-150",
        isDownloaded && "cursor-pointer hover:bg-surface-tertiary outline-hidden data-[focus]:shadow-focus",
        isError && "bg-danger-bg",
      )}
      {...(isDownloaded && onOpen
        ? {
            tabIndex: 0,
            onClick: onOpen,
            onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
              if (event.target !== event.currentTarget) return;
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onOpen();
              }
            },
          }
        : {})}
    >
      <div
        className={twMerge(
          "grid gap-3 pl-4 pr-3 items-center",
          compact ? "grid-cols-[minmax(0,1fr)_auto]" : "grid-cols-[minmax(0,1fr)_80px_200px_120px]",
        )}
      >
        <ModelHeading name={name} logo={logo} />

        {!compact && (
          <div className="text-left">
            {size && (
              <Text
                as="span"
                color="muted"
                opticalSize={14}
                className="text-[13px] font-[450] leading-[1.3] tabular-nums"
              >
                {size}
              </Text>
            )}
          </div>
        )}

        {!compact && (
          <div className="text-left min-w-0">
            {parameters && (
              <Text
                as="span"
                color="muted"
                opticalSize={14}
                className="text-[13px] font-[450] leading-[1.3] truncate block"
              >
                {parameters}
              </Text>
            )}
          </div>
        )}

        <div className="flex items-center justify-end gap-1.5">
          {isDownloading && (
            <Text
              as="span"
              opticalSize={14}
              className="text-[12px] font-[450] leading-[1.3] text-text-muted tabular-nums w-9 text-right"
            >
              {Math.round(progress)}%
            </Text>
          )}
          <DownloadActions
            status={status}
            isDownloading={isDownloading}
            isPaused={isPaused}
            isError={isError}
            onDownload={onDownload}
            onPause={onPause}
            onCancel={onCancel}
            onRetry={onRetry}
          />
          {isDownloaded && (
            <>
              <Tooltip content="Downloaded" side="top">
                <div className="flex items-center justify-center h-7 w-7 rounded border-[0.5px] border-border-outlined text-success">
                  <IconCheckStroke />
                </div>
              </Tooltip>
              <Button
                type="button"
                aria-label={`Delete ${name}`}
                title="Delete model (hold Shift to skip confirmation)"
                onClick={(event) => {
                  event.stopPropagation();
                  onDelete?.(event);
                }}
                className="flex items-center justify-center h-7 w-7 rounded cursor-pointer text-text-muted hover:text-danger bg-transparent hover:border-border-outlined-hover border-[0.5px] border-border-outlined outline-hidden data-[focus]:shadow-focus"
              >
                <Trash2 size={14} />
              </Button>
            </>
          )}
        </div>
      </div>

      {compact && (size || parameters) && (
        <div className="mt-2.5 pl-4 pr-3 flex items-center gap-1.5 flex-wrap">
          <ModelBadges parameters={parameters} size={size} />
        </div>
      )}

      {isDownloading && (
        <div className="mt-2.5 pl-4 pr-3">
          <div className="h-0.5 bg-border-default rounded overflow-clip">
            <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}
    </DataInteractive>
  );
}
