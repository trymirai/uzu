import type { FocusEvent, KeyboardEvent, MouseEvent } from "react";
import { cloneElement, useContext, useEffect, useId, useRef } from "react";
import { HoverPopoverContext } from "./provider";
import type { HoverPopoverProps } from "./types";

export function HoverPopover({ items, trigger, side = "bottom", align = "end" }: HoverPopoverProps) {
  const context = useContext(HoverPopoverContext);
  if (!context) throw new Error("HoverPopover must be rendered inside a HoverPopoverProvider.");
  const { show, hide, contains } = context;

  const id = useId();
  const triggerNodeRef = useRef<HTMLButtonElement | null>(null);

  // The panel lives in the provider, so it would outlive a trigger that unmounts while open.
  useEffect(() => () => hide(id), [hide, id]);

  const open = (options?: { focusPanel?: boolean }) => {
    if (triggerNodeRef.current) show({ id, triggerNode: triggerNodeRef.current, items, side, align }, options);
  };

  const mergedTrigger = cloneElement(trigger, {
    ref: (node: HTMLButtonElement | null) => {
      triggerNodeRef.current = node;
    },
    "aria-haspopup": "menu",
    onMouseEnter: () => open(),
    onMouseLeave: (event: MouseEvent<HTMLElement>) => {
      if (!contains(event.relatedTarget as Node | null)) hide(id);
    },
    onFocus: (event: FocusEvent<HTMLElement>) => {
      if (event.currentTarget.matches(":focus-visible")) open();
    },
    onBlur: (event: FocusEvent<HTMLElement>) => {
      if (!contains(event.relatedTarget as Node | null)) hide(id);
    },
    onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
      if (event.key === "Escape") hide(id);
      // The panel renders elsewhere in the tree, so Tab never reaches it; move
      // focus there explicitly.
      if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown") {
        event.preventDefault();
        open({ focusPanel: true });
      }
    },
  });

  return <div className="relative inline-flex">{mergedTrigger}</div>;
}
