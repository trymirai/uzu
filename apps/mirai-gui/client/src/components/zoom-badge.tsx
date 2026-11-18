import { useEffect, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";
import { MinusIcon, PlusIcon } from "@/components/icons/app-icons";
import { getPlatform } from "@/platform/platform-singleton";
import type { ZoomStep } from "@/platform/services/system-ui";

// The ends of the zoom steps in the desktop app (desktop/src/zoom.rs).
const MIN_ZOOM = 0.67;
const MAX_ZOOM = 1.75;
const SHOWN_FOR_MS = 2000;

/**
 * The interface zoom, shown for a moment after ⌘+, ⌘− or ⌘0, as in Claude and Codex: the level, a
 * step down, a step up and Reset. It stays while the pointer or focus is on it.
 *
 * It also publishes the level as `--app-zoom` on the page, for the few things that must keep their
 * place on screen however large the interface is drawn (the sidebar button beside the traffic
 * lights). The pill itself is one of them: it's drawn at the same size at every level.
 */
export function ZoomBadge() {
  const [level, setLevel] = useState(1);
  const [visible, setVisible] = useState(false);
  const held = useRef(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const scheduleHide = () => {
    clearTimeout(hideTimer.current);
    hideTimer.current = setTimeout(() => {
      if (!held.current) setVisible(false);
    }, SHOWN_FOR_MS);
  };

  useEffect(() => {
    const systemUi = getPlatform().systemUi;
    const publish = (next: number) => {
      setLevel(next);
      document.documentElement.style.setProperty("--app-zoom", String(next));
    };
    let changed = false;
    // The level the app opened at, without showing the pill: it appears only for a change.
    void systemUi.getZoom().then((initial) => {
      if (!changed) publish(initial);
    });
    const off = systemUi.onZoomChange((next) => {
      changed = true;
      publish(next);
      setVisible(true);
      scheduleHide();
    });
    return () => {
      off();
      clearTimeout(hideTimer.current);
    };
  }, []);

  const step = (direction: ZoomStep) => {
    void getPlatform()
      .systemUi.stepZoom(direction)
      .catch(() => undefined);
  };

  const hold = () => {
    held.current = true;
    clearTimeout(hideTimer.current);
  };
  const release = () => {
    held.current = false;
    scheduleHide();
  };

  const percent = Math.round(level * 100);

  return (
    // 16px from the window's corner: inside the content panel, as far from its edges as the panel is
    // from the window's.
    <div
      // Drawn at one size whatever the zoom: the browser scales everything by the level, this undoes it.
      style={{ zoom: 1 / level }}
      className={twMerge(
        "fixed top-4 right-4 z-50 flex h-9 items-center gap-0.5 rounded-[12px] bg-surface-popover pl-3 pr-1 shadow-popover",
        "origin-top-right transition-[opacity,scale] duration-150 ease-out motion-reduce:transition-none",
        visible ? "opacity-100 scale-100" : "pointer-events-none opacity-0 scale-[0.96] duration-100",
      )}
      // Out of the tab order while hidden (React 18 has no inert prop).
      ref={(element) => {
        if (element) element.inert = !visible;
      }}
      onPointerEnter={hold}
      onPointerLeave={release}
      onFocus={hold}
      onBlur={release}
    >
      <span role="status" className="min-w-10 text-[13px] font-[450] text-text-primary tabular-nums">
        {percent}%
      </span>
      <ZoomButton label="Zoom out" disabled={level <= MIN_ZOOM} onClick={() => step("out")}>
        <MinusIcon size={16} />
      </ZoomButton>
      <ZoomButton label="Zoom in" disabled={level >= MAX_ZOOM} onClick={() => step("in")}>
        <PlusIcon size={16} />
      </ZoomButton>
      <span aria-hidden className="mx-1 h-4 w-px bg-border-default" />
      <button
        type="button"
        disabled={level === 1}
        onClick={() => step("reset")}
        className="h-7 rounded-lg px-2 text-[13px] font-[450] text-text-primary outline-hidden transition-colors duration-150 ease-out hover:bg-selection-hover focus-visible:shadow-focus disabled:text-text-muted disabled:hover:bg-transparent"
      >
        Reset
      </button>
    </div>
  );
}

function ZoomButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="flex size-7 items-center justify-center rounded-lg text-text-muted outline-hidden transition-colors duration-150 ease-out hover:bg-selection-hover hover:text-text-primary focus-visible:shadow-focus disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-text-muted"
    >
      {children}
    </button>
  );
}
