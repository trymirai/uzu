import { createContext, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePopoverFloating } from "../shared/use-popover-floating";
import type { HoverPopoverContextValue, HoverPopoverPayload, HoverPopoverProviderProps } from "./types";
import { HoverPopoverPanel } from "./panel";

export const HoverPopoverContext = createContext<HoverPopoverContextValue | null>(null);

export function HoverPopoverProvider({ children }: HoverPopoverProviderProps) {
  const [active, setActive] = useState<HoverPopoverPayload | null>(null);
  const [visible, setVisible] = useState(false);
  const activeRef = useRef<HoverPopoverPayload | null>(null);
  const visibleRef = useRef(false);
  const panelNodeRef = useRef<HTMLDivElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const floating = usePopoverFloating({
    side: active?.side ?? "bottom",
    align: active?.align ?? "end",
    sideOffsetPx: 0,
  });

  useEffect(() => {
    activeRef.current = active;
    visibleRef.current = visible;
    floating.setTriggerRef(active?.triggerNode ?? null);
    restoreFocusRef.current = active?.triggerNode ?? null;
  }, [active, floating, visible]);

  const focusPanelPendingRef = useRef(false);

  const show = useCallback((payload: HoverPopoverPayload, options?: { focusPanel?: boolean }) => {
    activeRef.current = payload;
    focusPanelPendingRef.current = options?.focusPanel === true;
    setActive(payload);
    setVisible(true);
  }, []);

  useEffect(() => {
    if (!visible || !focusPanelPendingRef.current) return;
    focusPanelPendingRef.current = false;
    panelNodeRef.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus({ preventScroll: true });
  }, [visible, active]);

  const focusItemFrom = (current: HTMLElement, step: 1 | -1) => {
    const buttons = Array.from(
      panelNodeRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [],
    );
    if (buttons.length === 0) return;
    const index = buttons.findIndex((button) => button === current);
    buttons[(index + step + buttons.length) % buttons.length]?.focus();
  };

  const hide = useCallback((id: string) => {
    const current = activeRef.current;

    if (!current || current.id !== id) return;

    setVisible(false);
  }, []);

  const handleAfterLeave = useCallback(() => {
    if (!activeRef.current || visibleRef.current) return;

    activeRef.current = null;
    setActive(null);
    restoreFocusRef.current = null;
    floating.setTriggerRef(null);
  }, [floating]);

  // Moving between the trigger and its panel must not close the popover.
  const contains = useCallback((node: Node | null) => {
    const current = activeRef.current;
    if (!node || !current) return false;

    return Boolean(current.triggerNode.contains(node) || panelNodeRef.current?.contains(node));
  }, []);

  const contextValue = useMemo<HoverPopoverContextValue>(() => ({ show, hide, contains }), [contains, hide, show]);

  return (
    <HoverPopoverContext.Provider value={contextValue}>
      {children}
      {active && (
        <HoverPopoverPanel
          floatingStyle={floating.floatingStyle}
          setFloatingRef={(node) => {
            panelNodeRef.current = node;
            floating.setFloatingRef(node);
          }}
          open={visible}
          afterLeave={handleAfterLeave}
          close={() => hide(active.id)}
          items={active.items}
          side={active.side}
          align={active.align}
          sideOffsetPx={4}
          onMouseEnter={() => setVisible(true)}
          onMouseLeave={(event) => {
            if (contains(event.relatedTarget as Node | null)) return;
            setVisible(false);
          }}
          onFocusCapture={() => setVisible(true)}
          onBlurCapture={(event) => {
            if (contains(event.relatedTarget as Node | null)) return;
            setVisible(false);
          }}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              restoreFocusRef.current?.focus({ preventScroll: true });
              setVisible(false);
            } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              focusItemFrom(event.target as HTMLElement, event.key === "ArrowDown" ? 1 : -1);
            }
          }}
          restoreFocusRef={restoreFocusRef}
        />
      )}
    </HoverPopoverContext.Provider>
  );
}
