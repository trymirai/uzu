import { X } from "lucide-react";
import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from "@headlessui/react";
import { Fragment } from "react";
import { twMerge } from "tailwind-merge";
import { Button } from "../button";
import { Text } from "../typography";
import type { ModalProps } from "./types";

const SPRING_ENTER =
  "[transition:opacity_150ms_cubic-bezier(0.23,1,0.32,1),transform_150ms_cubic-bezier(0.23,1,0.32,1)]";
const SPRING_LEAVE =
  "[transition:opacity_100ms_cubic-bezier(0.23,1,0.32,1),transform_100ms_cubic-bezier(0.23,1,0.32,1)]";

export function Modal(props: ModalProps) {
  const {
    open,
    onClose,
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
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <TransitionChild
          as={Fragment}
          enter="duration-150 ease-out"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="duration-100 ease-in"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-overlay" />
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
            <DialogPanel
              className={twMerge(
                "rounded-lg bg-surface-elevated border border-border-default p-4 text-text-primary",
                "w-[468px]",
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <DialogTitle className="flex-1 min-w-0">
                  <Text as="span" color="primary" opticalSize={14} className="text-[18px] font-[450] leading-[28px]">
                    {title}
                  </Text>
                </DialogTitle>
                <button
                  type="button"
                  aria-label="Close"
                  onClick={onClose}
                  className="shrink-0 flex items-center justify-center size-7 rounded-md text-text-muted hover:text-text-primary hover:bg-tertiary-hover transition-colors duration-150 ease-out outline-none focus-visible:shadow-focus cursor-pointer"
                >
                  <X size={16} />
                </button>
              </div>
              {description && (
                <Text as="p" color="muted" opticalSize={20} className="mt-1.5 text-[15px] font-[450] leading-[1.6]">
                  {description}
                </Text>
              )}

              {children && <div className="mt-5">{children}</div>}

              <div className="mt-6 flex justify-end gap-2 flex-wrap">
                <Button kind="secondary" size="sm" onClick={handleSecondary}>
                  {secondaryLabel}
                </Button>
                {hasPrimary && (
                  <Button kind={primaryKind} size="sm" onClick={handlePrimary} disabled={primaryDisabled}>
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
