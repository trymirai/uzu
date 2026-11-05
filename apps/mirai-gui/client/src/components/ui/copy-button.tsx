import { IconButton } from "./icon-button";
import { useToast } from "./toast/use-toast";
import { IconSquareBehindSquare1 } from "@/components/icons/app-icons";
import { useEffect, useId, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";

type CopyButtonProps = {
  onCopy: () => Promise<void>;
  className?: string;
};

const COPIED_FEEDBACK_MS = 1200;

// The same plain button as the other actions under an answer. `active:scale-100` cancels the shared
// icon button's own press shrink, which made the row jump on a tap.
const BUTTON_CLASSES =
  "relative size-8 flex items-center justify-center p-0 rounded-lg text-label-muted transition-colors duration-150 ease-out hover:bg-bg-hover hover:text-label-title active:scale-100";

// Both icons sit in the same spot at full size and cross-fade, with 2px of blur bridging the two so
// the eye reads one icon changing rather than two objects trading places. CSS transitions, so a
// second click mid-swap simply turns it back. Each layer keeps its own GPU layer and rests at
// blur(0px), never `none`: WebKit drops the layer when the filter goes away and the icon snaps by a
// subpixel, which read as a jump at the end of the swap.
const LAYER =
  "absolute inset-0 flex items-center justify-center will-change-[opacity,filter] transition-[opacity,filter] duration-200 ease-out motion-reduce:transition-none";
const SHOWN = "opacity-100 [filter:blur(0px)]";
const HIDDEN = "opacity-0 [filter:blur(2px)]";

/** The circle-check, filled: a solid disc with the check cut out of it. */
function CircleCheckFilled({ size = 16 }: { size?: number }) {
  const maskId = useId();
  return (
    <svg aria-hidden width={size} height={size} viewBox="0 0 24 24" fill="none">
      <mask id={maskId} maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">
        <rect width="24" height="24" fill="#fff" />
        <path d="M15 9.5L10.5 15L8.5 13" stroke="#000" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </mask>
      <circle cx="12" cy="12" r="10" fill="currentColor" mask={`url(#${maskId})`} />
    </svg>
  );
}

export function CopyButton({ onCopy, className }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const toast = useToast();
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => clearTimeout(timer.current ?? undefined), []);

  const handleCopy = async () => {
    try {
      await onCopy();
      setCopied(true);
      clearTimeout(timer.current ?? undefined);
      timer.current = setTimeout(() => setCopied(false), COPIED_FEEDBACK_MS);
    } catch {
      toast.error("Copy failed. Please try again.");
    }
  };

  return (
    <IconButton
      aria-label={copied ? "Copied" : "Copy"}
      onClick={handleCopy}
      className={twMerge(BUTTON_CLASSES, className)}
    >
      <span className="relative block size-4">
        <span aria-hidden className={twMerge(LAYER, copied ? HIDDEN : SHOWN)}>
          <IconSquareBehindSquare1 size={16} />
        </span>
        {/* The check keeps the bright, hovered colour while it shows, whether or not the pointer
            is still on the button. */}
        <span aria-hidden className={twMerge(LAYER, "text-label-title", copied ? SHOWN : HIDDEN)}>
          <CircleCheckFilled size={16} />
        </span>
      </span>
    </IconButton>
  );
}
