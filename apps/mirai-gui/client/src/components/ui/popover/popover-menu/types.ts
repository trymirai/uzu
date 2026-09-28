import type { PopoverAlign, PopoverItem, PopoverSide, PopoverTrigger } from "../shared/types";

export type PopoverMenuItem = PopoverItem;
export type PopoverMenuSide = PopoverSide;
export type PopoverMenuAlign = PopoverAlign;

export type PopoverMenuProps = {
  items: ReadonlyArray<PopoverMenuItem>;
  trigger: PopoverTrigger;
  side?: PopoverMenuSide;
  align?: PopoverMenuAlign;
  sideOffsetPx?: number;
  className?: string;
  itemClassName?: string;
};
