import { Transition } from "@headlessui/react";
import { useParams } from "@tanstack/react-router";
import React, { useEffect, useLayoutEffect, useRef } from "react";
import { useChatStore } from "@/stores/use-chat-store";
import { Loader, LoaderIcon } from "@/components/loader";
import { Message } from "./message";

type MessageListProps = {
  isNew?: boolean;
  isLoading?: boolean;
  scrollToMessageId?: string | null;
  onScrolled?: () => void;
  isUiStreaming?: boolean;
  onMessageModelSelect?: (messageId: string, modelId: string, modelName: string) => void;
  loadingMessageId?: string | null;
  canceledMessageId?: string | null;
  isModelLoading?: boolean;
  hasResidentModel?: boolean;
};

const MessageListComponent: React.FC<MessageListProps> = ({
  isNew,
  isLoading,
  scrollToMessageId,
  onScrolled,
  isUiStreaming,
  onMessageModelSelect,
  loadingMessageId,
  canceledMessageId,
  isModelLoading,
  hasResidentModel,
}) => {
  const messages = useChatStore((s) => s.messages);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const lastScrolledIdRef = useRef<string | null>(null);
  const autoStickRef = useRef<boolean>(true);
  const { chatId } = useParams({ from: "/chat/$chatId" });

  const scrollToBottom = (behavior: ScrollBehavior = "auto") => {
    const container = scrollContainerRef.current;
    if (container) {
      if (behavior === "smooth") {
        container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
      } else {
        container.scrollTop = container.scrollHeight;
      }
    } else {
      if (behavior === "smooth") {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      } else {
        messagesEndRef.current?.scrollIntoView();
      }
    }
  };

  const didInitialScrollRef = useRef<boolean>(false);
  useLayoutEffect(() => {
    didInitialScrollRef.current = false;
    autoStickRef.current = true;
  }, [chatId]);
  useLayoutEffect(() => {
    if (!didInitialScrollRef.current && messages.length > 0) {
      scrollToBottom("auto");
      didInitialScrollRef.current = true;
    }
  }, [messages.length, chatId]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const threshold = 8;

    const handleScroll = () => {
      const atBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - threshold;
      autoStickRef.current = atBottom;
    };

    const handleLoad = () => {
      if (autoStickRef.current) {
        container.scrollTop = container.scrollHeight;
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    container.addEventListener("load", handleLoad, true);

    const observer = new MutationObserver(() => {
      if (autoStickRef.current) {
        container.scrollTop = container.scrollHeight;
      }
    });
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    const resizeObserver = new ResizeObserver(() => {
      if (autoStickRef.current) {
        container.scrollTop = container.scrollHeight;
      }
    });
    resizeObserver.observe(container);

    handleScroll();

    return () => {
      container.removeEventListener("scroll", handleScroll);
      container.removeEventListener("load", handleLoad, true);
      observer.disconnect();
      resizeObserver.disconnect();
    };
  }, [chatId]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    let rafId = 0;
    const tick = () => {
      if (autoStickRef.current && isLoading && !isUiStreaming) {
        container.scrollTop = container.scrollHeight;
      }
      rafId = requestAnimationFrame(tick);
    };

    if (isLoading && !isUiStreaming) {
      rafId = requestAnimationFrame(tick);
    }

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isUiStreaming, isLoading, chatId]);

  useEffect(() => {
    if (!scrollToMessageId) return;
    if (lastScrolledIdRef.current === scrollToMessageId) return;

    const container = scrollContainerRef.current;
    if (!container) return;

    const tryScroll = () => {
      const el = itemRefs.current[scrollToMessageId];
      if (!el) {
        return;
      }
      container.scrollTo({ top: el.offsetTop, behavior: "smooth" });
      lastScrolledIdRef.current = scrollToMessageId;
      onScrolled?.();
    };

    requestAnimationFrame(tryScroll);
  }, [scrollToMessageId, onScrolled]);

  const lastAssistantIndex = messages.at(-1)?.sender === "assistant" ? messages.length - 1 : -1;

  const showWaiting = !hasResidentModel && Boolean(isLoading) && !loadingMessageId;
  const isBusy = Boolean(isModelLoading || isLoading);
  const loaderText = showWaiting ? "Waiting for the model to be ready…" : "Generating reply…";

  return (
    <div className="relative flex flex-1 flex-col overflow-hidden h-full">
      <div ref={scrollContainerRef} className="absolute inset-0 flex flex-col gap-4 overflow-y-auto scrollbar-hide">
        {messages.length > 0 ? (
          messages.map((message, idx) => {
            const isLastAssistant = idx === lastAssistantIndex;
            const isLoadingThisMessage = loadingMessageId === message.id;

            const hasVersions = Array.isArray(message.versions) && (message.versions?.length || 0) > 0;
            const activeVersion = hasVersions
              ? message.versions?.[message.currentVersionIndex ?? message.versions.length - 1]
              : undefined;
            const activeParsed = hasVersions ? activeVersion?.output?.text?.parsed : message.output?.text?.parsed;
            const activeText = hasVersions ? activeVersion?.text : message.text;

            const hasVisibleParsed = Boolean(
              (activeParsed?.chainOfThought && activeParsed.chainOfThought.length > 0) ||
                (activeParsed?.response && activeParsed.response.length > 0) ||
                (activeText && activeText.length > 0),
            );

            const showInlineLoader = isLoadingThisMessage && !hasVisibleParsed;

            const showAssistantLoader = !loadingMessageId && isLastAssistant && isBusy;

            return (
              <Transition
                key={message.id}
                appear
                show={true}
                enter="transition-opacity duration-200 ease-out"
                enterFrom="opacity-0"
                enterTo="opacity-100"
              >
                <div
                  ref={(el) => {
                    itemRefs.current[message.id] = el;
                  }}
                  className={isLastAssistant ? "flex flex-col min-h-8" : undefined}
                >
                  {showAssistantLoader ? (
                    <div className="flex items-center gap-2 text-left text-[15px] leading-[18px] text-label-muted">
                      <Loader text={loaderText} />
                    </div>
                  ) : null}
                  {showInlineLoader ? (
                    <div className="flex items-center gap-2 text-left text-[15px] leading-[18px] text-label-muted">
                      <Loader text={loaderText} />
                    </div>
                  ) : null}
                  <Message
                    key={`${message.id}_${message.currentVersionIndex ?? 0}_${message.versions?.length ?? 0}`}
                    isLoading={isLoadingThisMessage}
                    chatId={chatId}
                    isLast={idx === lastAssistantIndex}
                    isUiStreaming={Boolean(isUiStreaming && isLastAssistant)}
                    isCanceled={canceledMessageId === message.id}
                    onModelSelect={onMessageModelSelect}
                    {...message}
                  />
                </div>
              </Transition>
            );
          })
        ) : (
          <div className="flex gap-2 items-center justify-center h-full relative">
            {isNew ? (
              <>
                <span className="text-label-muted text-[13px] font-[350]">
                  Start your new private and local conversation
                </span>
                <LoaderIcon />
              </>
            ) : (
              <Loader />
            )}
          </div>
        )}
        {lastAssistantIndex === -1 && isBusy && !loadingMessageId && (
          <div className="flex items-center gap-2 text-left text-[15px] leading-[18px] text-label-muted">
            <Loader text={loaderText} />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export const MessageList = React.memo(MessageListComponent);
