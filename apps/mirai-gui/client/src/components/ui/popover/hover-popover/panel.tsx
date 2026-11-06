import { Transition, TransitionChild } from "@headlessui/react";
import type {
  CSSProperties,
  FocusEvent as ReactFocusEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  RefCallback,
  RefObject,
} from "react";
import { Fragment, useState } from "react";
import { twMerge } from "tailwind-merge";
import { PopoverItemLabel } from "../shared/item-label";
import { PopoverSurface } from "../shared/surface";
import type { PopoverItem } from "../shared/types";
import { getOriginClassName } from "../shared/utils";
import type { HoverPopoverAlign, HoverPopoverSide } from "./types";

type HoverPopoverPanelProps = {
  floatingStyle: CSSProperties;
  setFloatingRef: RefCallback<HTMLDivElement>;
  open: boolean;
  afterLeave?: () => void;
  close: () => void;
  items: ReadonlyArray<PopoverItem>;
  footerItems?: ReadonlyArray<PopoverItem>;
  maxVisibleItems?: number;
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

// A row's height (min-h-8), for limiting the list to a number of rows.
const ROW_HEIGHT_PX = 32;

const ENTER = "transition-[opacity,transform] duration-200 ease-spring motion-reduce:transition-none";
const LEAVE = "transition-[opacity,transform] duration-[140ms] ease-spring motion-reduce:transition-none";

export function HoverPopoverPanel({
  floatingStyle,
  setFloatingRef,
  open,
  afterLeave,
  close,
  items,
  footerItems,
  maxVisibleItems,
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

  const [moreBelow, setMoreBelow] = useState(false);

  const renderItem = (item: PopoverItem) => (
    <button
      key={item.id}
      type="button"
      role="menuitem"
      disabled={!open || item.disabled === true}
      className={twMerge(
        "flex min-h-8 items-center gap-2.5 w-full px-2 py-1.5 rounded-lg bg-transparent border-none outline-hidden text-left transition-colors duration-100 ease-out hover:bg-selection-hover focus:bg-selection-hover",
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
  );

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
      className={twMerge("z-50 outline-hidden", open ? "pointer-events-auto" : "pointer-events-none")}
    >
      <Transition appear show={open} as={Fragment} afterLeave={afterLeave}>
        <TransitionChild
          as="div"
          enter={ENTER}
          enterFrom="opacity-0 scale-95"
          enterTo="opacity-100 scale-100"
          leave={LEAVE}
          leaveFrom="opacity-100 scale-100"
          leaveTo="opacity-0 scale-95"
          className={getOriginClassName(side, align)}
        >
          <PopoverSurface open side={side} align={align} animated={false}>
            {/* Long lists scroll inside the panel (a fixed number of rows when asked for) instead of
                running off the window. When more rows wait below, half of the next one shows and
                fades out, so the list reads as scrollable; the fade goes once the end is reached.
                Footer rows stay put under a hairline. */}
            <div
              ref={(node) => {
                if (node) setMoreBelow(node.scrollHeight - node.clientHeight - node.scrollTop > 1);
              }}
              onScroll={(event) => {
                const node = event.currentTarget;
                setMoreBelow(node.scrollHeight - node.clientHeight - node.scrollTop > 1);
              }}
              className={twMerge(
                "max-h-[min(20rem,calc(100vh-6rem))] overflow-y-auto overscroll-contain",
                moreBelow && "[mask-image:linear-gradient(to_bottom,black_calc(100%-1.5rem),transparent)]",
              )}
              style={maxVisibleItems ? { maxHeight: `${(maxVisibleItems + 0.5) * ROW_HEIGHT_PX}px` } : undefined}
            >
              {items.map(renderItem)}
            </div>
            {footerItems?.length ? (
              <>
                <div aria-hidden className="mx-2 my-1 h-px bg-border-default" />
                {footerItems.map(renderItem)}
              </>
            ) : null}
          </PopoverSurface>
        </TransitionChild>
      </Transition>
    </div>
  );
}
