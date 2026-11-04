import { IconCrossSmall } from "@central-icons-react/round-outlined-radius-3-stroke-2";
import {
  Button as HeadlessButton,
  Dialog,
  DialogPanel,
  DialogTitle,
  Transition,
  TransitionChild,
} from "@headlessui/react";
import { Fragment } from "react";
import { twMerge } from "tailwind-merge";
import { Button } from "../button";
import { Text } from "../typography";
import { SURFACE_BUTTON_CLASSNAME, SURFACE_CLOSE_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME } from "./constants";
import type { ModalProps } from "./types";

const SPRING_ENTER =
  "[transition:opacity_150ms_cubic-bezier(0.23,1,0.32,1),transform_150ms_cubic-bezier(0.23,1,0.32,1)]";
const SPRING_LEAVE =
  "[transition:opacity_100ms_cubic-bezier(0.23,1,0.32,1),transform_100ms_cubic-bezier(0.23,1,0.32,1)]";

export function Modal(props: ModalProps) {
  const {
    open,
    onClose,
    initialFocus,
    title,
    description,
    children,
    primaryLabel,
    primaryKind = "primary",
    onPrimary,
    primaryDisabled = false,
    secondaryLabel = "Cancel",
    onSecondary,
  } = props;

  const handlePrimary = onPrimary ?? onClose;
  const handleSecondary = onSecondary ?? onClose;
  const hasPrimary = Boolean(primaryLabel);

  return (
    <Transition appear show={open} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose} initialFocus={initialFocus}>
        <TransitionChild
          as={Fragment}
          enter="duration-150 ease-out"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="duration-100 ease-in"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-overlay motion-reduce:transition-none" />
        </TransitionChild>

        <div className="fixed inset-0 flex items-center justify-center p-4">
          <TransitionChild
            as={Fragment}
            enter={SPRING_ENTER}
            enterFrom="opacity-0 scale-[0.96]"
            enterTo="opacity-100 scale-100"
            leave={SPRING_LEAVE}
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-[0.96]"
          >
            {/* The same lifted surface as menus and toasts, so a dialog is one more layer of the
                app rather than a different kind of window. */}
            <DialogPanel className="w-[440px] max-w-full rounded-2xl bg-surface-popover shadow-popover p-5 text-text-primary motion-reduce:transition-none">
              <div className="flex items-start justify-between gap-3">
                <DialogTitle className="flex-1 min-w-0 pt-0.5">
                  <Text
                    as="span"
                    color="primary"
                    opticalSize={16}
                    className="block text-[16px] font-[500] leading-[1.35] text-balance"
                  >
                    {title}
                  </Text>
                </DialogTitle>
                <HeadlessButton
                  type="button"
                  aria-label="Close"
                  onClick={onClose}
                  className={twMerge(SURFACE_CLOSE_CLASSNAME, "-mr-1 -mt-0.5")}
                >
                  <IconCrossSmall size={18} aria-hidden />
                </HeadlessButton>
              </div>
              {description && (
                <Text
                  as="p"
                  color="secondary"
                  opticalSize={14}
                  className="mt-1.5 text-[14px] font-[400] leading-[1.5] text-pretty [overflow-wrap:anywhere]"
                >
                  {description}
                </Text>
              )}

              {children && <div className="mt-4">{children}</div>}

              <div className="mt-6 flex justify-end gap-2 flex-wrap">
                <Button
                  kind="secondary"
                  size="sm"
                  onClick={handleSecondary}
                  className={twMerge(SURFACE_BUTTON_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME)}
                >
                  {secondaryLabel}
                </Button>
                {hasPrimary && (
                  <Button
                    kind={primaryKind}
                    size="sm"
                    onClick={handlePrimary}
                    disabled={primaryDisabled}
                    className={SURFACE_BUTTON_CLASSNAME}
                  >
                    {primaryLabel}
                  </Button>
                )}
              </div>
            </DialogPanel>
          </TransitionChild>
        </div>
      </Dialog>
    </Transition>
  );
}
