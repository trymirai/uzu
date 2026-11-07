import type { ReactElement, ReactNode } from "react";
import type { PopoverAlign, PopoverItem, PopoverSide } from "../shared/types";

export type HoverPopoverItem = PopoverItem;
export type HoverPopoverSide = PopoverSide;
export type HoverPopoverAlign = PopoverAlign;

export type HoverPopoverProviderProps = {
  children: ReactNode;
};

export type HoverPopoverProps = {
  items: ReadonlyArray<HoverPopoverItem>;
  /** Rows under the list that stay put while it scrolls, e.g. "More models". */
  footerItems?: ReadonlyArray<HoverPopoverItem>;
  /** How many rows of `items` show in full before the list scrolls; half of the next one peeks. */
  maxVisibleItems?: number;
  trigger: ReactElement;
  side?: HoverPopoverSide;
  align?: HoverPopoverAlign;
};

export type HoverPopoverPayload = {
  id: string;
  triggerNode: HTMLButtonElement;
  items: ReadonlyArray<HoverPopoverItem>;
  footerItems?: ReadonlyArray<HoverPopoverItem>;
  maxVisibleItems?: number;
  side: HoverPopoverSide;
  align: HoverPopoverAlign;
};

export type HoverPopoverContextValue = {
  show: (payload: HoverPopoverPayload, options?: { focusPanel?: boolean }) => void;
  hide: (id: string) => void;
  contains: (node: Node | null) => boolean;
};
