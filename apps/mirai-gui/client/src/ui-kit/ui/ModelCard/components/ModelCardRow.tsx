import type { KeyboardEvent } from "react";
import { twMerge } from "tailwind-merge";
import { IconCheckStroke } from "../../IconCheckStroke";
import { Tooltip } from "../../Tooltip";
import { Text } from "../../Typography";
import type { ModelCardProps } from "../types";
import { deriveModelCardState } from "../modelCardState";
import { DownloadActions } from "./DownloadActions";
import { ModelBadges } from "./ModelBadges";
import { ModelHeading } from "./ModelHeading";
import { ModelOptions } from "./ModelOptions";

export function ModelCardRow({
  name,
  logo,
  parameters,
  size,
  isThinking = false,
  isCompressed = false,
  state,
  onDownload,
  onPause,
  onCancel,
  onDelete,
  onRetry,
  onOpen,
  compact = false,
  quickDelete = false,
}: ModelCardProps) {
  const { status, isDownloading, isDownloaded, isPaused, isError, progress } = deriveModelCardState(state);

  return (
    <div
      className={twMerge(
        "relative py-2.5 transition-colors duration-150",
        isDownloaded && "cursor-pointer hover:bg-surface-tertiary outline-none focus-visible:shadow-focus",
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
              <Tooltip content="Installed" side="top">
                <div className="flex items-center justify-center h-7 w-7 rounded border-[0.5px] border-border-outlined text-success">
                  <IconCheckStroke />
                </div>
              </Tooltip>
              <div className="relative" onClick={(event) => event.stopPropagation()}>
                <ModelOptions quickDelete={quickDelete} onDelete={onDelete} />
              </div>
            </>
          )}
        </div>
      </div>

      {compact && (size || parameters || isCompressed || isThinking) && (
        <div className="mt-2.5 pl-4 pr-3 flex items-center gap-1.5 flex-wrap">
          <ModelBadges parameters={parameters} size={size} isThinking={isThinking} isCompressed={isCompressed} />
        </div>
      )}

      {isDownloading && (
        <div className="mt-2.5 pl-4 pr-3">
          <div className="h-0.5 bg-border-default rounded overflow-hidden">
            <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
