import { type KeyboardEvent, type PointerEvent, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";
import {
  SIDEBAR_DEFAULT_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
  useSidebarStore,
} from "@/stores/use-sidebar-store";

// Dragged this far past the minimum width, the sidebar lets go and hides; dragged back out from the
// edge, it opens again. As in Claude and Codex.
const COLLAPSE_BELOW = SIDEBAR_MIN_WIDTH - 60;
const KEY_STEP = 16;

/**
 * The sidebar's right edge, as a handle: drag to resize, drag fully left to hide, drag out from the
 * window edge to bring it back, double-click to reset the width. Also a focusable separator: the
 * arrow keys resize it. A thin line shows where it is while hovered or held.
 */
export function SidebarResizer({ onDraggingChange }: { onDraggingChange: (dragging: boolean) => void }) {
  const isOpen = useSidebarStore((s) => s.isOpen);
  const width = useSidebarStore((s) => s.width);
  const setOpen = useSidebarStore((s) => s.setOpen);
  const setWidth = useSidebarStore((s) => s.setWidth);
  const drag = useRef<{ startX: number; startWidth: number } | null>(null);
  const [active, setActive] = useState(false);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { startX: event.clientX, startWidth: isOpen ? width : 0 };
    setActive(true);
    onDraggingChange(true);
    document.body.style.cursor = "col-resize";
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    const next = drag.current.startWidth + (event.clientX - drag.current.startX);
    if (next < COLLAPSE_BELOW) {
      if (isOpen) setOpen(false);
      return;
    }
    if (!isOpen) setOpen(true);
    setWidth(next);
  };

  const endDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setActive(false);
    onDraggingChange(false);
    document.body.style.cursor = "";
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? KEY_STEP : -KEY_STEP;
    if (!isOpen) {
      if (delta > 0) setOpen(true);
      return;
    }
    if (width + delta < SIDEBAR_MIN_WIDTH) setOpen(false);
    else setWidth(width + delta);
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize sidebar"
      aria-valuemin={0}
      aria-valuemax={SIDEBAR_MAX_WIDTH}
      aria-valuenow={isOpen ? width : 0}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={() => {
        setOpen(true);
        setWidth(SIDEBAR_DEFAULT_WIDTH);
      }}
      onKeyDown={onKeyDown}
      className="group absolute inset-y-0 -right-1.5 z-40 flex w-3 cursor-col-resize justify-center outline-hidden touch-none"
    >
      {/* Shown after a short rest on hover (so passing the edge doesn't flash it), at once while held
          or focused from the keyboard. Stops short of the ends, clear of the content panel's rounded
          corners. */}
      <span
        aria-hidden
        className={twMerge(
          "my-6 w-0.5 rounded-full bg-blue opacity-0 transition-opacity duration-150 ease-out group-hover:opacity-100 group-hover:delay-150 group-focus-visible:opacity-100",
          active && "opacity-100 delay-0",
        )}
      />
    </div>
  );
}
