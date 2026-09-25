import { Transition, TransitionChild } from "@headlessui/react";
import type {
  CSSProperties,
  FocusEvent as ReactFocusEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  RefCallback,
  RefObject,
} from "react";
import { Fragment } from "react";
import { twMerge } from "tailwind-merge";
import { PopoverItemLabel } from "../../shared/components/PopoverItemLabel";
import { PopoverSurface } from "../../shared/components/PopoverSurface";
import type { PopoverItem } from "../../shared/types";
import { getOriginClassName } from "../../shared/utils";
import type { HoverPopoverAlign, HoverPopoverSide } from "../types";

type HoverPopoverPanelProps = {
  floatingStyle: CSSProperties;
  setFloatingRef: RefCallback<HTMLDivElement>;
  open: boolean;
  afterLeave?: () => void;
  close: () => void;
  items: ReadonlyArray<PopoverItem>;
  side: HoverPopoverSide;
  align: HoverPopoverAlign;
  sideOffsetPx: number;
  onMouseEnter?: () => void;
  onMouseLeave?: (event: ReactMouseEvent<HTMLDivElement>) => void;
  onFocusCapture?: () => void;
  onBlurCapture?: (event: ReactFocusEvent<HTMLDivElement>) => void;
  onKeyDown?: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
  restoreFocusRef?: RefObject<HTMLElement | null>;
};

const ENTER = "transition-[opacity,transform] duration-[120ms] ease-spring";
const LEAVE = "transition-[opacity,transform] duration-[80ms] ease-spring";

export function HoverPopoverPanel({
  floatingStyle,
  setFloatingRef,
  open,
  afterLeave,
  close,
  items,
  side,
  align,
  sideOffsetPx,
  onMouseEnter,
  onMouseLeave,
  onFocusCapture,
  onBlurCapture,
  onKeyDown,
  restoreFocusRef,
}: HoverPopoverPanelProps) {
  const wrapperStyle = {
    ...floatingStyle,
    ...(side === "top" ? { paddingBottom: sideOffsetPx } : { paddingTop: sideOffsetPx }),
  } satisfies CSSProperties;

  return (
    <div
      ref={setFloatingRef}
      style={wrapperStyle}
      {...(open ? { onMouseEnter } : {})}
      {...(open ? { onMouseLeave } : {})}
      {...(open ? { onFocusCapture } : {})}
      {...(open ? { onBlurCapture } : {})}
      {...(open ? { onKeyDown } : {})}
      role="menu"
      className={twMerge("z-50 outline-none", open ? "pointer-events-auto" : "pointer-events-none")}
    >
      <Transition appear show={open} as={Fragment} afterLeave={afterLeave}>
        <TransitionChild
          as="div"
          enter={ENTER}
          enterFrom="opacity-0 scale-90"
          enterTo="opacity-100 scale-100"
          leave={LEAVE}
          leaveFrom="opacity-100 scale-100"
          leaveTo="opacity-0 scale-90"
          className={getOriginClassName(side, align)}
        >
          <PopoverSurface open side={side} align={align} animated={false}>
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                role="menuitem"
                disabled={!open || item.disabled === true}
                className={twMerge(
                  "flex items-center gap-2 w-full px-2.5 py-2 rounded-md bg-transparent border-none outline-none text-left hover:bg-surface-tertiary focus:bg-surface-tertiary",
                  item.disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
                  item.danger ? "text-red-500" : "text-text-primary",
                )}
                onClick={(event) => {
                  if (item.disabled) return;
                  restoreFocusRef?.current?.focus({ preventScroll: true });
                  item.onClick?.(event);
                  close();
                }}
              >
                {item.icon && <span className="shrink-0">{item.icon}</span>}
                <PopoverItemLabel label={item.label} />
              </button>
            ))}
          </PopoverSurface>
        </TransitionChild>
      </Transition>
    </div>
  );
}
