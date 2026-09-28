import { IconButton } from "./icon-button";
import { useToast } from "./toast/use-toast";
import { Transition } from "@headlessui/react";
import { Check, Copy } from "lucide-react";
import { Fragment, useState } from "react";
import { twMerge } from "tailwind-merge";

type CopyButtonProps = {
  onCopy: () => Promise<void>;
  className?: string;
};

const COPIED_FEEDBACK_MS = 1000;

const BUTTON_CLASSES =
  "relative min-w-8 min-h-8 flex items-center justify-center p-[6px] rounded-[5px] text-label-muted dark:text-label-muted-dark hover:bg-card-hover dark:hover:bg-card-hover-dark hover:text-label-title dark:hover:text-label-title-dark active:scale-100";

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
    <IconButton variant="pill" aria-label="Copy" onClick={handleCopy} className={twMerge(BUTTON_CLASSES, className)}>
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
        <Copy className="w-[14px] h-[14px] absolute" />
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
        <Check className="w-[14px] h-[14px] absolute" />
      </Transition>
    </IconButton>
  );
}
