import { autoUpdate, offset, shift, useFloating, type Placement } from "@floating-ui/react-dom";
import type { CSSProperties, RefCallback } from "react";
import type { PopoverAlign, PopoverSide } from "../types";

export type PopoverFloatingConfig = {
  side: PopoverSide;
  align: PopoverAlign;
  sideOffsetPx: number;
};

export type PopoverFloatingResult = {
  floatingStyle: CSSProperties;
  setTriggerRef: RefCallback<HTMLButtonElement>;
  setFloatingRef: RefCallback<HTMLDivElement>;
};

const getPlacement = (side: PopoverSide, align: PopoverAlign): Placement => {
  const resolvedSide = side === "top" ? "top" : "bottom";
  const resolvedAlign = align === "start" ? "start" : "end";

  return `${resolvedSide}-${resolvedAlign}`;
};

export function usePopoverFloating({ side, align, sideOffsetPx }: PopoverFloatingConfig): PopoverFloatingResult {
  const placement = getPlacement(side, align);

  const { refs, floatingStyles } = useFloating({
    strategy: "fixed",
    placement,
    whileElementsMounted: autoUpdate,
    middleware: [offset(sideOffsetPx), shift({ padding: 8 })],
  });

  return {
    floatingStyle: floatingStyles,
    setTriggerRef: refs.setReference,
    setFloatingRef: refs.setFloating,
  };
}
