import { Menu, MenuButton } from "@headlessui/react";
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  ReactElement,
} from "react";
import { Fragment, cloneElement, useCallback, useEffect, useRef } from "react";
import { usePopoverFloating } from "../shared/use-popover-floating";
import type { PopoverMenuProps } from "./types";
import { PopoverMenuPanel } from "./panel";

const renderTrigger = (trigger: PopoverMenuProps["trigger"], open: boolean): ReactElement =>
  typeof trigger === "function" ? trigger(open) : trigger;

type PopoverMenuContentProps = {
  menuOpen: boolean;
  close: () => void;
  trigger: PopoverMenuProps["trigger"];
  items: PopoverMenuProps["items"];
  side: NonNullable<PopoverMenuProps["side"]>;
  align: NonNullable<PopoverMenuProps["align"]>;
  className: string | undefined;
  itemClassName: string | undefined;
  setTriggerRef: (node: HTMLButtonElement | null) => void;
  setFloatingRef: (node: HTMLDivElement | null) => void;
  floatingStyle: CSSProperties;
};

function PopoverMenuContent({
  menuOpen,
  close,
  trigger,
  items,
  side,
  align,
  className,
  itemClassName,
  setTriggerRef,
  setFloatingRef,
  floatingStyle,
}: PopoverMenuContentProps) {
  const openMethodRef = useRef<"keyboard" | "pointer" | null>(null);

  const setMergedItemsRef = useCallback(
    (node: HTMLDivElement | null) => {
      setFloatingRef(node);

      if (node === null || !menuOpen || openMethodRef.current !== "keyboard") {
        return;
      }

      const firstEnabledItem = node.querySelector<HTMLButtonElement>("button:not(:disabled)");

      firstEnabledItem?.focus({ preventScroll: true });
    },
    [menuOpen, setFloatingRef],
  );

  useEffect(() => {
    if (!menuOpen) {
      openMethodRef.current = null;
    }
  }, [menuOpen]);

  const triggerElement = renderTrigger(trigger, menuOpen);
  const triggerOnKeyDownCapture = triggerElement.props.onKeyDownCapture as
    | ((event: ReactKeyboardEvent<HTMLElement>) => void)
    | undefined;
  const triggerOnPointerDownCapture = triggerElement.props.onPointerDownCapture as
    | ((event: ReactPointerEvent<HTMLElement>) => void)
    | undefined;

  const mergedTrigger = cloneElement(triggerElement, {
    onKeyDownCapture: (event: ReactKeyboardEvent<HTMLElement>) => {
      triggerOnKeyDownCapture?.(event);
      if (event.defaultPrevented) return;
      if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown" || event.key === "ArrowUp") {
        openMethodRef.current = "keyboard";
      }
    },
    onPointerDownCapture: (event: ReactPointerEvent<HTMLElement>) => {
      triggerOnPointerDownCapture?.(event);
      if (event.defaultPrevented) return;
      openMethodRef.current = "pointer";
    },
  });

  return (
    <>
      <MenuButton ref={setTriggerRef} as={Fragment}>
        {mergedTrigger}
      </MenuButton>
      {menuOpen ? (
        <PopoverMenuPanel
          floatingStyle={floatingStyle}
          setFloatingRef={setMergedItemsRef}
          menuOpen={menuOpen}
          close={close}
          items={items}
          side={side}
          align={align}
          className={className}
          itemClassName={itemClassName}
        />
      ) : null}
    </>
  );
}

export function PopoverMenu({
  items,
  trigger,
  side = "bottom",
  align = "end",
  sideOffsetPx,
  className,
  itemClassName,
}: PopoverMenuProps) {
  const resolvedSideOffsetPx = sideOffsetPx ?? 4;
  const floating = usePopoverFloating({
    side,
    align,
    sideOffsetPx: resolvedSideOffsetPx,
  });

  return (
    <Menu as="div" className="relative inline-flex">
      {({ open: menuOpen, close }) => (
        <PopoverMenuContent
          menuOpen={menuOpen}
          close={close}
          trigger={trigger}
          items={items}
          side={side}
          align={align}
          className={className}
          itemClassName={itemClassName}
          setTriggerRef={floating.setTriggerRef}
          setFloatingRef={floating.setFloatingRef}
          floatingStyle={floating.floatingStyle}
        />
      )}
    </Menu>
  );
}
