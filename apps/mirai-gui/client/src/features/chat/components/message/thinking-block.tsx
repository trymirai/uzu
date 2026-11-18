import { Transition } from "@headlessui/react";
import { ChevronRightIcon } from "lucide-react";
import { memo, useCallback, useEffect, useRef, useState, type UIEvent } from "react";
import { twMerge } from "tailwind-merge";
import { TextShimmer } from "@/components/ui/text-shimmer";
import { MarkdownRenderer } from "./markdown-renderer";

type Props = {
  text: string;
  reasoningInProgress: boolean;
  isThinking?: boolean;
  /** Tool calls made within this run of thinking; named in the summary once it is done. */
  toolCalls?: number;
};

const FADE_RAMP_PX = 32;
const MAX_FADE_HEIGHT_PX = 42;
const END_FADE_HEIGHT_PX = MAX_FADE_HEIGHT_PX / 2;

export const ThinkingBlock = memo(function ThinkingBlock({
  text,
  reasoningInProgress,
  isThinking = reasoningInProgress,
  toolCalls = 0,
}: Props) {
  const [isReasoningVisible, setIsReasoningVisible] = useState(false);
  // How long it thought, measured while you watched; reopened chats just say it thought.
  const [seconds, setSeconds] = useState<number | null>(null);
  const thinkingSinceRef = useRef<number | null>(null);
  useEffect(() => {
    if (isThinking) {
      thinkingSinceRef.current ??= performance.now();
      return;
    }
    if (thinkingSinceRef.current === null) return;
    setSeconds(Math.max(1, Math.round((performance.now() - thinkingSinceRef.current) / 1000)));
    thinkingSinceRef.current = null;
  }, [isThinking]);
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
      // A line of text, not a box: the cat beside the answer already shows that it is thinking.
      className="group mb-1 block w-full text-left [--fade:var(--color-background)] outline-hidden data-[focus]:shadow-focus rounded-md"
    >
      <span className="flex h-7 items-center gap-1 text-[15px] font-[400] leading-[1.3] text-label-muted transition-colors duration-150 ease-out group-hover:text-label-title">
        {isThinking ? (
          <TextShimmer duration={1.6} className="[--base-color:var(--color-label-muted)]">
            Thinking…
          </TextShimmer>
        ) : (
          <span key="done" className="animate-uzu-swap tabular-nums">
            {seconds !== null ? `Thought for ${seconds} s` : "Thought"}
            {toolCalls > 0 ? ` · ${toolCalls} ${toolCalls === 1 ? "tool call" : "tool calls"}` : ""}
          </span>
        )}
        <ChevronRightIcon
          aria-hidden
          className={twMerge(
            "size-3.5 shrink-0 transition-transform duration-250 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none",
            isReasoningVisible && "rotate-90",
          )}
        />
      </span>
      {/* Opens on the same curve as the chevron, the text fading in as the height grows. It stays
          mounted while closed, so opening never has to render the whole reasoning in its first
          frame, which is what made it stutter. */}
      <Transition
        show={isReasoningVisible}
        unmount={false}
        enter="transition-[grid-template-rows,opacity] duration-250 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none"
        enterFrom="grid-rows-[0fr] opacity-0"
        enterTo="grid-rows-[1fr] opacity-100"
        leave="transition-[grid-template-rows,opacity] duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none"
        leaveFrom="grid-rows-[1fr] opacity-100"
        leaveTo="grid-rows-[0fr] opacity-0"
      >
        <div className="grid">
          <div className="min-h-0 min-w-0 overflow-clip">
            <div className="relative contain-layout">
              <div
                ref={setReasoningScrollElement}
                onScroll={onReasoningScroll}
                className="pb-2 pt-0.5 text-left max-h-[180px] overflow-y-auto overscroll-y-auto thin-scrollbar"
              >
                <MarkdownRenderer
                  content={text || ""}
                  useOneFontSize={true}
                  className="text-[15px] leading-[1.6] text-label-muted [&_p]:mb-2 [&_p]:mt-0"
                />
              </div>
              <div
                data-reasoning-fade="top"
                aria-hidden
                style={{
                  height: Math.min(MAX_FADE_HEIGHT_PX * fades.top, fades.viewportHeight / 2),
                  opacity: (2 / 3) * fades.top,
                }}
                className="pointer-events-none absolute top-0 inset-x-0 z-10 bg-linear-to-b from-(--fade) to-transparent"
              />
              <div
                data-reasoning-fade="bottom"
                aria-hidden
                style={{
                  height: Math.min(END_FADE_HEIGHT_PX + END_FADE_HEIGHT_PX * fades.bottom, fades.viewportHeight / 2),
                  opacity: (1 + fades.bottom) / 3,
                }}
                className="pointer-events-none absolute bottom-0 inset-x-0 z-10 bg-linear-to-b from-transparent to-(--fade)"
              />
            </div>
          </div>
        </div>
      </Transition>
    </button>
  );
});
