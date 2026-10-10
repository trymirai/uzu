import { Transition } from "@headlessui/react";
import { ChevronDownIcon } from "lucide-react";
import { memo, useCallback, useEffect, useRef, useState, type UIEvent } from "react";
import { twMerge } from "tailwind-merge";
import { ThinkingBubbleIcon } from "@/components/icons/thinking-bubble";
import { MarkdownRenderer } from "./markdown-renderer";

type Props = { text: string; reasoningInProgress: boolean; isThinking?: boolean };

const FADE_RAMP_PX = 32;
const MAX_FADE_HEIGHT_PX = 42;
const END_FADE_HEIGHT_PX = MAX_FADE_HEIGHT_PX / 2;

export const ThinkingBlock = memo(function ThinkingBlock({
  text,
  reasoningInProgress,
  isThinking = reasoningInProgress,
}: Props) {
  const [isReasoningVisible, setIsReasoningVisible] = useState(false);
  const wasReasoningInProgressRef = useRef(false);
  const userToggledRef = useRef(false);

  const [reasoningScrollElement, setReasoningScrollElement] = useState<HTMLDivElement | null>(null);

  const reasoningPinnedRef = useRef(true);

  const reasoningAutoScrollingRef = useRef(false);
  const [fades, setFades] = useState({ top: 0, bottom: 0, viewportHeight: 0 });

  const updateFades = useCallback((el: HTMLDivElement) => {
    const viewportHeight = el.clientHeight;
    const overflow = Math.max(0, el.scrollHeight - viewportHeight);
    const hiddenTop = Math.max(0, Math.min(el.scrollTop, overflow));
    const top = Math.min(hiddenTop / FADE_RAMP_PX, 1);
    const bottom = Math.min((overflow - hiddenTop) / FADE_RAMP_PX, 1);
    setFades((current) =>
      current.top === top && current.bottom === bottom && current.viewportHeight === viewportHeight
        ? current
        : { top, bottom, viewportHeight },
    );
  }, []);

  const onReasoningScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    updateFades(el);
    if (reasoningAutoScrollingRef.current) return;
    reasoningPinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  const updateReasoningScroll = useCallback(() => {
    const el = reasoningScrollElement;
    if (!el) return;
    if (reasoningPinnedRef.current) {
      reasoningAutoScrollingRef.current = true;
      el.scrollTop = el.scrollHeight;

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          reasoningAutoScrollingRef.current = false;
        });
      });
    }
    updateFades(el);
  }, [reasoningScrollElement, updateFades]);

  useEffect(() => {
    if (isReasoningVisible) updateReasoningScroll();
  }, [text, isReasoningVisible, updateReasoningScroll]);

  useEffect(() => {
    const el = reasoningScrollElement;
    if (!isReasoningVisible || !el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateReasoningScroll);
    observer.observe(el);
    if (el.firstElementChild) observer.observe(el.firstElementChild);
    return () => observer.disconnect();
  }, [isReasoningVisible, reasoningScrollElement, updateReasoningScroll]);

  useEffect(() => {
    if (wasReasoningInProgressRef.current && !reasoningInProgress) {
      setIsReasoningVisible(false);
    } else if (reasoningInProgress && text && !userToggledRef.current) {
      setIsReasoningVisible(true);
    }
    wasReasoningInProgressRef.current = reasoningInProgress;
  }, [reasoningInProgress, text]);

  return (
    <button
      data-message-reasoning
      aria-expanded={isReasoningVisible}
      onClick={() => {
        userToggledRef.current = true;
        setIsReasoningVisible((v) => !v);
      }}
      className={twMerge(
        "group mb-1 px-3 pt-2 border border-cell-border w-full rounded-[8px] transition-colors duration-150",
        "hover:[background-color:rgba(0,0,0,0.01)] dark:hover:[background-color:rgba(255,255,255,0.01)]",
      )}
    >
      <div className="flex pb-2 items-center justify-between p-0 group hover:bg-transparent text-label-title">
        <span className="flex items-center gap-2">
          <ThinkingBubbleIcon className="w-[14px] h-[14px]" />

          <span className="text-[13px] font-[400] leading-[130%]">{isThinking ? "Thinking..." : "Thinking"}</span>
        </span>
        <ChevronDownIcon
          className={twMerge(
            "w-[12px] h-[12px] text-label-muted transition-transform transition-colors duration-150",
            isReasoningVisible && "rotate-180",
          )}
        />
      </div>
      <Transition
        show={isReasoningVisible}
        appear
        enter="transition-[grid-template-rows] duration-200 ease-in-out motion-reduce:transition-none"
        enterFrom="grid-rows-[0fr]"
        enterTo="grid-rows-[1fr]"
        leave="transition-[grid-template-rows] duration-200 ease-in-out motion-reduce:transition-none"
        leaveFrom="grid-rows-[1fr]"
        leaveTo="grid-rows-[0fr]"
      >
        <div className="grid">
          <div className="min-h-0 min-w-0 overflow-clip">
            <div className="relative contain-layout">
              <div
                ref={setReasoningScrollElement}
                onScroll={onReasoningScroll}
                className="pb-2 text-left max-h-[180px] overflow-y-auto overscroll-y-auto thin-scrollbar"
              >
                <MarkdownRenderer
                  content={text || ""}
                  useOneFontSize={true}
                  className="text-[12px] leading-[150%] font-mono text-label-muted [&_p]:mb-1 [&_p]:mt-1"
                />
              </div>
              <div
                data-reasoning-fade="top"
                aria-hidden
                style={{
                  height: Math.min(MAX_FADE_HEIGHT_PX * fades.top, fades.viewportHeight / 2),
                  opacity: (2 / 3) * fades.top,
                }}
                className="pointer-events-none absolute top-0 inset-x-0 z-10 [background:linear-gradient(180deg,#FFFFFF_0%,rgba(255,255,255,0)_100%)] dark:[background:linear-gradient(180deg,#0A0A0A_0%,rgba(10,10,10,0)_100%)]"
              />
              <div
                data-reasoning-fade="bottom"
                aria-hidden
                style={{
                  height: Math.min(END_FADE_HEIGHT_PX + END_FADE_HEIGHT_PX * fades.bottom, fades.viewportHeight / 2),
                  opacity: (1 + fades.bottom) / 3,
                }}
                className="pointer-events-none absolute bottom-0 inset-x-0 z-10 [background:linear-gradient(180deg,rgba(255,255,255,0)_0%,#FFFFFF_100%)] dark:[background:linear-gradient(180deg,rgba(10,10,10,0)_0%,#0A0A0A_100%)]"
              />
            </div>
          </div>
        </div>
      </Transition>
    </button>
  );
});
