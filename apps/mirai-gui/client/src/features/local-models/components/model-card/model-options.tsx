import { Ellipsis, Trash2 } from "lucide-react";
import type { MouseEvent } from "react";
import { HoverPopover } from "@/components/ui/popover/hover-popover";

type ModelOptionsProps = {
  quickDelete?: boolean;
  onDelete?: (event: MouseEvent<HTMLElement>) => void;
};

export function ModelOptions({ quickDelete = false, onDelete }: ModelOptionsProps) {
  if (quickDelete) {
    return (
      <button
        type="button"
        aria-label="Delete model"
        onClick={onDelete}
        className="flex items-center justify-center h-7 w-7 rounded cursor-pointer text-danger bg-transparent hover:border-border-outlined-hover border-[0.5px] border-border-outlined outline-none focus-visible:shadow-focus"
      >
        <Trash2 size={14} />
      </button>
    );
  }

  return (
    <HoverPopover
      side="top"
      align="end"
      trigger={
        <button
          type="button"
          aria-label="Model options"
          className="group/chip flex items-center justify-center h-7 w-7 rounded cursor-pointer text-text-muted hover:text-text-primary bg-transparent hover:border-border-outlined-hover border-[0.5px] border-border-outlined outline-none focus-visible:shadow-focus"
        >
          <Ellipsis size={16} />
        </button>
      }
      items={[
        {
          id: "delete",
          label: "Delete",
          icon: <Trash2 size={14} />,
          danger: true,
          onClick: onDelete,
        },
      ]}
    />
  );
}
