import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import { syncAriaDescribedBy } from "./utils";

export function useTooltipAria({
  wrapperRef,
  tooltipId,
  visible,
}: {
  wrapperRef: RefObject<HTMLSpanElement | null>;
  tooltipId: string;
  visible: boolean;
}) {
  const describedElementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const wrapperNode = wrapperRef.current;

    if (!wrapperNode) return undefined;

    const activeElement = document.activeElement;
    const focusedDescendant =
      activeElement instanceof HTMLElement && activeElement !== wrapperNode && wrapperNode.contains(activeElement)
        ? activeElement
        : null;

    const previousElement = describedElementRef.current;

    if (previousElement && previousElement !== focusedDescendant) {
      syncAriaDescribedBy(previousElement, tooltipId, false);
    }

    if (!visible || !focusedDescendant) {
      describedElementRef.current = null;
      return undefined;
    }

    syncAriaDescribedBy(focusedDescendant, tooltipId, true);
    describedElementRef.current = focusedDescendant;

    return () => {
      syncAriaDescribedBy(describedElementRef.current, tooltipId, false);
    };
  }, [tooltipId, visible, wrapperRef]);
}
