import { IconButton } from "./icon-button";
import { useToast } from "./toast/use-toast";
import { Transition } from "@headlessui/react";
import { IconCheckmark1Small, IconSquareBehindSquare1 } from "@central-icons-react/round-outlined-radius-3-stroke-2";
import { Fragment, useState } from "react";
import { twMerge } from "tailwind-merge";

type CopyButtonProps = {
  onCopy: () => Promise<void>;
  className?: string;
};

const COPIED_FEEDBACK_MS = 1000;

const BUTTON_CLASSES =
  "relative size-8 flex items-center justify-center p-0 rounded-lg text-label-muted transition-[background-color,color,scale] duration-150 ease-out hover:bg-black/[0.06] hover:text-label-title dark:hover:bg-white/[0.08] active:scale-[0.96]";

export function CopyButton({ onCopy, className }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const toast = useToast();

  const handleCopy = async () => {
    try {
      await onCopy();
      setCopied(true);
      setTimeout(() => setCopied(false), COPIED_FEEDBACK_MS);
    } catch {
      toast.error("Copy failed. Please try again.");
    }
  };

  return (
    <IconButton aria-label="Copy" onClick={handleCopy} className={twMerge(BUTTON_CLASSES, className)}>
      <Transition
        as={Fragment}
        show={!copied}
        enter="transition-opacity duration-500"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-500"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <IconSquareBehindSquare1 size={16} aria-hidden className="absolute" />
      </Transition>
      <Transition
        as={Fragment}
        show={copied}
        enter="transition-opacity duration-500"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-500"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <IconCheckmark1Small size={16} aria-hidden className="absolute text-success-strong" />
      </Transition>
    </IconButton>
  );
}
