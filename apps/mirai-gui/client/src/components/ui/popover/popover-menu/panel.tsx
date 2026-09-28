import { MenuItem, MenuItems } from "@headlessui/react";
import type { CSSProperties, RefCallback } from "react";
import { twMerge } from "tailwind-merge";
import { PopoverItemLabel } from "../shared/item-label";
import { PopoverSurface } from "../shared/surface";
import type { PopoverItem } from "../shared/types";
import type { PopoverMenuAlign, PopoverMenuSide } from "./types";

type PopoverMenuPanelProps = {
  floatingStyle: CSSProperties;
  setFloatingRef: RefCallback<HTMLDivElement>;
  menuOpen: boolean;
  close: () => void;
  items: ReadonlyArray<PopoverItem>;
  side: PopoverMenuSide;
  align: PopoverMenuAlign;
  className?: string;
  itemClassName?: string;
};

export function PopoverMenuPanel({
  floatingStyle,
  setFloatingRef,
  menuOpen,
  close,
  items,
  side,
  align,
  className,
  itemClassName,
}: PopoverMenuPanelProps) {
  return (
    <MenuItems
      ref={setFloatingRef}
      style={floatingStyle}
      className={twMerge("z-50 outline-none", menuOpen ? "pointer-events-auto" : undefined)}
    >
      <PopoverSurface open={menuOpen} side={side} align={align} className={className}>
        {items.map((item) => (
          <MenuItem
            key={item.id}
            as="button"
            type="button"
            disabled={item.disabled ?? false}
            className={({ focus, disabled }) =>
              twMerge(
                "flex items-center gap-2 w-full px-2.5 py-2 rounded-md bg-transparent border-none outline-none text-left",
                focus ? "bg-surface-tertiary" : undefined,
                disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
                item.danger ? "text-red-500" : "text-text-primary",
                itemClassName,
              )
            }
            onClick={(event) => {
              if (item.disabled) return;
              close();
              item.onClick?.(event);
            }}
          >
            {item.icon && <span className="shrink-0">{item.icon}</span>}
            <PopoverItemLabel label={item.label} />
          </MenuItem>
        ))}
      </PopoverSurface>
    </MenuItems>
  );
}
