import { flip, offset, shift, useFloating } from "@floating-ui/react-dom";
import type { ReferenceType } from "@floating-ui/react-dom";
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { TooltipContext } from "./context";
import type { TooltipContextValue, TooltipPayload, TooltipProviderProps } from "./types";
import { autoUpdateWithDetach } from "./utils";
import { TooltipBubble } from "./bubble";

export function TooltipProvider({ children }: TooltipProviderProps) {
  const [activeTooltip, setActiveTooltip] = useState<TooltipPayload | null>(null);
  const tooltipId = useId();

  const whileElementsMounted = useCallback(
    (reference: ReferenceType, floating: HTMLElement, update: () => void) =>
      autoUpdateWithDetach(reference, floating, update, () => {
        setActiveTooltip((current) => (current && current.reference === reference ? null : current));
      }),
    [],
  );

  const { refs, floatingStyles } = useFloating({
    placement: activeTooltip?.side ?? "top",
    strategy: "fixed",
    whileElementsMounted,
    middleware: [offset(8), flip(), shift({ padding: 8 })],
  });

  useEffect(() => {
    refs.setReference(activeTooltip?.reference ?? null);
  }, [activeTooltip, refs]);

  const show = useCallback((payload: TooltipPayload) => {
    setActiveTooltip(payload);
  }, []);

  const hide = useCallback((id: string) => {
    setActiveTooltip((current) => (current?.id === id ? null : current));
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
          tooltipId={tooltipId}
          content={activeTooltip.content}
          floatingStyles={floatingStyles}
          setFloating={refs.setFloating}
        />
      )}
    </TooltipContext.Provider>
  );
}
