import React from "react";
import { DataInteractive } from "@headlessui/react";
import { formatDistanceToNow } from "date-fns";
import { ChatsIcon } from "@/components/icons/chats-icon";
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
        "rounded-lg overflow-clip bg-background cursor-pointer transition-colors border-[0.5px] border-cell-border",
        !selectMode && "outline-hidden data-[focus]:shadow-focus",
        selectMode ? "hover:bg-card-hover" : "hover:bg-card-hover",
        selected ? "bg-bg-modal border-button-border" : "",
      )}
    >
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-3 p-3 w-full">
        <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
          {selectMode ? (
            <span onClick={(e) => e.stopPropagation()}>
              <Checkbox
                checked={checked}
                onChange={() => onCheckedChange?.()}
                size="sm"
                aria-label={`Select ${title}`}
              />
            </span>
          ) : (
            <ChatsIcon className="w-5 h-5 text-label-title" />
          )}
        </div>
        <div className="min-w-0">
          <h3 className="text-[15px] leading-[150%] font-[350] text-label-title overflow-clip text-ellipsis whitespace-nowrap">
            {title}
          </h3>
        </div>
        <div className="flex-shrink-0 ml-2">
          <span className="text-xs text-label-muted font-mono whitespace-nowrap">{timeAgo}</span>
        </div>
      </div>
    </DataInteractive>
  );
};

export default ChatCard;
