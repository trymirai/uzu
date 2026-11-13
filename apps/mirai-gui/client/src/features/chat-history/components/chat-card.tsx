import React from "react";
import { DataInteractive } from "@headlessui/react";
import { formatDistanceToNow } from "date-fns";
import { Checkbox } from "@/components/ui/checkbox";
import { twMerge } from "tailwind-merge";

type ChatCardProps = {
  id: string;
  title: string;
  updatedAt: number;
  onClick: () => void;
  selected?: boolean;
  selectMode?: boolean;
  checked?: boolean;
  onCheckedChange?: () => void;
};

const ChatCard: React.FC<ChatCardProps> = ({
  id,
  title,
  updatedAt,
  onClick,
  selected = false,
  selectMode = false,
  checked = false,
  onCheckedChange,
}) => {
  const timeAgo = formatDistanceToNow(updatedAt, { addSuffix: true });

  return (
    <DataInteractive
      as="div"
      key={id}
      onClick={onClick}
      // In select mode the checkbox is the focusable control, so the card must not
      // also be a button: that would nest a control inside a control.
      {...(selectMode
        ? {}
        : {
            role: "button",
            tabIndex: 0,
            onKeyDown: (event: React.KeyboardEvent<HTMLDivElement>) => {
              if (event.key !== "Enter" && event.key !== " ") return;
              event.preventDefault();
              onClick();
            },
          })}
      className={twMerge(
        "rounded-xl bg-surface-soft cursor-pointer transition-colors duration-150 ease-out",
        !selectMode && "outline-hidden data-[focus]:bg-surface-tertiary",
        // Two steps, as in the menus: the pointer lifts a card a little, choosing it a step more.
        selected ? "bg-selection-selected" : "hover:bg-surface-tertiary",
      )}
    >
      {/* The title alone leads; the checkbox takes the leading slot only while selecting. */}
      <div
        className={twMerge(
          "grid items-center gap-3 h-14 px-4 w-full",
          selectMode ? "grid-cols-[auto_1fr_auto]" : "grid-cols-[1fr_auto]",
        )}
      >
        {selectMode ? (
          // A flex box, not an inline span: inline, the box sat on the text baseline, below centre.
          <div className="flex size-4 shrink-0 items-center justify-center" onClick={(e) => e.stopPropagation()}>
            <Checkbox checked={checked} onChange={() => onCheckedChange?.()} size="sm" aria-label={`Select ${title}`} />
          </div>
        ) : null}
        <div className="min-w-0">
          <h3 className="text-[14px] leading-[1.35] font-[450] text-label-title overflow-clip text-ellipsis whitespace-nowrap">
            {title}
          </h3>
        </div>
        <div className="flex-shrink-0 ml-2">
          <span className="text-[13px] text-label-muted tabular-nums whitespace-nowrap">{timeAgo}</span>
        </div>
      </div>
    </DataInteractive>
  );
};

export default ChatCard;
