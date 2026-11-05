import { flip, offset, shift, useFloating } from "@floating-ui/react-dom";
import type { ReferenceType } from "@floating-ui/react-dom";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { TooltipContext } from "./context";
import type { TooltipContextValue, TooltipPayload, TooltipProviderProps } from "./types";
import { autoUpdateWithDetach } from "./utils";
import { TooltipBubble } from "./bubble";

// The first tooltip waits so it doesn't flash on a passing cursor; once one has been shown, its
// neighbours open at once and without an entrance, until the pointer has been away a moment.
const OPEN_DELAY_MS = 400;
const WARM_FOR_MS = 300;
export const PANEL_GAP_PX = 6;

export function TooltipProvider({ children }: TooltipProviderProps) {
  const [activeTooltip, setActiveTooltip] = useState<TooltipPayload | null>(null);
  const [instant, setInstant] = useState(false);
  const tooltipId = useId();
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const warmUntilRef = useRef(0);
  const visibleRef = useRef(false);

  const whileElementsMounted = useCallback(
    (reference: ReferenceType, floating: HTMLElement, update: () => void) =>
      autoUpdateWithDetach(reference, floating, update, () => {
        setActiveTooltip((current) => (current && current.reference === reference ? null : current));
      }),
    [],
  );

  const { refs, floatingStyles, isPositioned } = useFloating({
    placement: activeTooltip
      ? activeTooltip.align === "start"
        ? (`${activeTooltip.side}-start` as const)
        : activeTooltip.side
      : "top",
    strategy: "fixed",
    // Lets `isPositioned` fall back to false between tooltips, so each one waits for its own place.
    open: activeTooltip !== null,
    whileElementsMounted,
    // Panels sit as close to their trigger as the hover menus do (PANEL_GAP_PX); label tips a touch further.
    middleware: [offset(activeTooltip?.variant === "panel" ? PANEL_GAP_PX : 8), flip(), shift({ padding: 8 })],
  });

  // Before paint, so a tooltip is never measured against the previous trigger (it showed up out of
  // place on the first hover of a new one).
  useLayoutEffect(() => {
    refs.setReference(activeTooltip?.reference ?? null);
  }, [activeTooltip, refs]);

  useEffect(() => {
    visibleRef.current = activeTooltip !== null;
  }, [activeTooltip]);
  useEffect(() => () => clearTimeout(openTimerRef.current ?? undefined), []);

  const show = useCallback((payload: TooltipPayload) => {
    clearTimeout(openTimerRef.current ?? undefined);
    const warm = visibleRef.current || performance.now() < warmUntilRef.current;
    if (warm) {
      setInstant(true);
      setActiveTooltip(payload);
      return;
    }
    openTimerRef.current = setTimeout(() => {
      setInstant(false);
      setActiveTooltip(payload);
    }, payload.delay ?? OPEN_DELAY_MS);
  }, []);

  const hide = useCallback((id: string) => {
    clearTimeout(openTimerRef.current ?? undefined);
    setActiveTooltip((current) => {
      if (current?.id !== id) return current;
      warmUntilRef.current = performance.now() + WARM_FOR_MS;
      return null;
    });
  }, []);

  const contextValue = useMemo<TooltipContextValue>(
    () => ({
      activeId: activeTooltip?.id ?? null,
      tooltipId,
      show,
      hide,
    }),
    [activeTooltip?.id, hide, show, tooltipId],
  );

  return (
    <TooltipContext.Provider value={contextValue}>
      {children}
      {activeTooltip && (
        <TooltipBubble
          key={activeTooltip.id}
          side={activeTooltip.side}
          align={activeTooltip.align ?? "center"}
          variant={activeTooltip.variant ?? "label"}
          instant={instant}
          tooltipId={tooltipId}
          content={activeTooltip.content}
          floatingStyles={floatingStyles}
          positioned={isPositioned}
          setFloating={refs.setFloating}
        />
      )}
    </TooltipContext.Provider>
  );
}
