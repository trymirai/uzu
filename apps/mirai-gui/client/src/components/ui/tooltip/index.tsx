import { useCallback, useContext, useId, useLayoutEffect, useRef } from "react";
import { TooltipContext } from "./context";
import { useTooltipAria } from "./use-tooltip-aria";
import type { TooltipContextValue, TooltipProps } from "./types";

function TooltipShared({ content, side = "top", children, context }: TooltipProps & { context: TooltipContextValue }) {
  const instanceId = useId();
  const wrapperRef = useRef<HTMLSpanElement | null>(null);
  const visible = context.activeId === instanceId;

  useTooltipAria({ wrapperRef, tooltipId: context.tooltipId, visible });

  const show = useCallback(() => {
    if (!wrapperRef.current) return;

    context.show({
      id: instanceId,
      reference: wrapperRef.current,
      content,
      side,
    });
  }, [content, context, instanceId, side]);

  const hide = useCallback(() => {
    context.hide(instanceId);
  }, [context, instanceId]);

  const hideRef = useRef(hide);
  hideRef.current = hide;
  useLayoutEffect(() => {
    return () => {
      hideRef.current();
    };
  }, []);

  return (
    <span
      ref={wrapperRef}
      className="inline-flex"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onPointerDown={(event) => {
        if (event.button === 0) hide();
      }}
    >
      {children}
    </span>
  );
}

export function Tooltip(props: TooltipProps) {
  const context = useContext(TooltipContext);
  if (!context) {
    // The bubble lives in the provider, so without one the trigger renders bare and
    // the tip silently disappears. Warn instead of failing quietly on reuse.
    if (import.meta.env.DEV) {
      console.warn("Tooltip rendered without a TooltipProvider; the tooltip will not appear.");
    }
    return <>{props.children}</>;
  }
  return <TooltipShared {...props} context={context} />;
}
