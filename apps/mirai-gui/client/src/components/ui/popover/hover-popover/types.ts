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
  trigger: ReactElement;
  side?: HoverPopoverSide;
  align?: HoverPopoverAlign;
};

export type HoverPopoverPayload = {
  id: string;
  triggerNode: HTMLButtonElement;
  items: ReadonlyArray<HoverPopoverItem>;
  side: HoverPopoverSide;
  align: HoverPopoverAlign;
};

export type HoverPopoverContextValue = {
  show: (payload: HoverPopoverPayload, options?: { focusPanel?: boolean }) => void;
  hide: (id: string) => void;
  contains: (node: Node | null) => boolean;
};
