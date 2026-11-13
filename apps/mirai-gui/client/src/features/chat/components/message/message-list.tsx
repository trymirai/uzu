import { useParams } from "@tanstack/react-router";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useChatStore } from "@/stores/use-chat-store";
import { Loader } from "@/components/loader";
import { AssistantCat, PendingStatus } from "./assistant-cat";
import { Message } from "./message";
import { NewChatCat } from "./new-chat-cat";
import { UzuBirdHost } from "./uzu-bird";

type MessageListProps = {
  isNew?: boolean;
  isLoading?: boolean;
  scrollToMessageId?: string | null;
  onScrolled?: () => void;
  isUiStreaming?: boolean;
  onMessageModelSelect?: (messageId: string, modelId: string, modelName: string) => void;
  onEditMessage?: (messageId: string, text: string, onSaved: () => void) => Promise<void>;
  canEditMessages?: boolean;
  loadingMessageId?: string | null;
  canceledMessageId?: string | null;
  isModelLoading?: boolean;
  /** The composer's text, which the empty chat's cat watches being typed. */
  draft?: string;
  /** When a send last failed for want of a model (the empty chat's cat reacts to it). */
  missingModelAt?: number;
};

const MessageListComponent: React.FC<MessageListProps> = ({
  isNew,
  isLoading,
  scrollToMessageId,
  onScrolled,
  isUiStreaming,
  onMessageModelSelect,
  onEditMessage,
  canEditMessages,
  loadingMessageId,
  canceledMessageId,
  isModelLoading,
  draft = "",
  missingModelAt = 0,
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

  // Messages already there when a chat opens are simply shown; ones that arrive while you watch
  // come into focus. A new chat starts with nothing on record, so everything in it arrives.
  const shownRef = useRef<{ chatId: string; ids: Set<string> | null }>({ chatId, ids: null });
  if (shownRef.current.chatId !== chatId) shownRef.current = { chatId, ids: null };
  if (!shownRef.current.ids && (isNew || messages.length > 0)) {
    shownRef.current.ids = new Set(isNew ? [] : messages.map((m) => m.id));
  }
  const arrived = (id: string) => !shownRef.current.ids?.has(id);

  // With room in the margin the cat hangs outside the column, so answers line up with the
  // composer and your messages; on narrow windows it sits inline beside the answer.
  const [catHangs, setCatHangs] = useState(false);
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const column = parseFloat(getComputedStyle(container).getPropertyValue("--chat-width")) || Infinity;
      setCatHangs((container.clientWidth - column) / 2 >= 48);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    measure();
    return () => observer.disconnect();
  }, [chatId]);

  const lastAssistantIndex = messages.at(-1)?.sender === "assistant" ? messages.length - 1 : -1;

  const isBusy = Boolean(isModelLoading || isLoading);
  // "Loading the model" only when the engine says it is loading one; otherwise the prompt is being
  // read. Guessing from what the interface last knew flashed "loading" on every first message.
  const pendingPhase = isModelLoading ? "loadingModel" : "reading";

  return (
    <div className="relative flex flex-1 flex-col overflow-clip h-full">
      <div
        ref={scrollContainerRef}
        data-cat-hang={catHangs || undefined}
        className="absolute inset-0 flex flex-col overflow-y-auto overscroll-y-contain scrollbar-hide pt-[var(--chat-list-top,2rem)]"
        style={{ paddingInline: "max(1.25rem, calc((100% - var(--chat-width)) / 2))" }}
      >
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
            const activeTranscript = hasVersions ? activeVersion?.output?.transcript : message.output?.transcript;

            const hasVisibleParsed = Boolean(
              (activeParsed?.chainOfThought && activeParsed.chainOfThought.length > 0) ||
              (activeParsed?.response && activeParsed.response.length > 0) ||
              activeTranscript?.length ||
              (activeText && activeText.length > 0),
            );

            const showInlineLoader = isLoadingThisMessage && !hasVisibleParsed;

            // Only before the answer has anything to show. Once thinking or text has arrived the
            // message tells its own state; on fast models the engine can still read as busy between
            // thinking and the first words, which brought "Reading your message" back mid-answer.
            const showAssistantLoader = !loadingMessageId && isLastAssistant && isBusy && !hasVisibleParsed;

            return (
              <div
                key={message.id}
                ref={(el) => {
                  itemRefs.current[message.id] = el;
                }}
                className={[
                  "shrink-0",
                  isLastAssistant ? "flex flex-col min-h-8" : "",
                  idx > 0 ? "mt-6" : "",
                  arrived(message.id) ? "animate-uzu-in" : "",
                ].join(" ")}
              >
                <Message
                  key={`${message.id}_${message.currentVersionIndex ?? 0}_${message.versions?.length ?? 0}`}
                  isLoading={isLoadingThisMessage}
                  chatId={chatId}
                  isLast={idx === lastAssistantIndex}
                  isUiStreaming={Boolean(isUiStreaming && isLastAssistant)}
                  isCanceled={canceledMessageId === message.id}
                  onModelSelect={onMessageModelSelect}
                  onEdit={onEditMessage}
                  canEdit={canEditMessages}
                  pendingPhase={showAssistantLoader || showInlineLoader ? pendingPhase : null}
                  {...message}
                />
              </div>
            );
          })
        ) : isNew ? (
          // A new chat sits in the middle of the page with the composer right under it.
          <div className="flex flex-col items-center justify-end h-full relative pb-7">
            {/* Keyed by chat, so each new chat gets its own greeting. */}
            <NewChatCat key={chatId} draft={draft} missingModelAt={missingModelAt} />
          </div>
        ) : (
          <div className="flex gap-2 items-center justify-center h-full relative">
            <Loader />
          </div>
        )}
        {lastAssistantIndex === -1 && isBusy && !loadingMessageId && (
          <div className="relative mt-6 flex gap-3 animate-uzu-in in-data-cat-hang:block">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center in-data-cat-hang:absolute in-data-cat-hang:-left-10 in-data-cat-hang:top-0">
              <AssistantCat phase={pendingPhase} live />
            </div>
            <PendingStatus phase={pendingPhase} />
          </div>
        )}
        {/* Room for the composer and the gap under it, which the list runs behind, plus 24px above. */}
        <div
          ref={messagesEndRef}
          className="shrink-0"
          style={{ height: messages.length > 0 ? "calc(var(--composer-h, 0px) + 1.25rem + 1.5rem)" : "1.5rem" }}
        />
      </div>
      <UzuBirdHost />
    </div>
  );
};

export const MessageList = React.memo(MessageListComponent);
