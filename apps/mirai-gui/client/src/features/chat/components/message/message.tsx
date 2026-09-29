import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import { writeItemsWithFocus } from "@/utils/clipboard";
import { Transition } from "@headlessui/react";
import { ChevronDownIcon } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";
import type { MessageVersion } from "@/types/message";
import { useChatStore } from "@/stores/use-chat-store";
import { attachmentStorage } from "../../services/attachment-storage";
import { ModelMenuIcon } from "@/components/icons/model-menu-icon";
import { ThinkingBubbleIcon } from "@/components/icons/thinking-bubble";
import { AttachedFilesDisplay } from "../composer/attached-files-display";
import { CopyButton } from "@/components/ui/copy-button";
import { MarkdownRenderer } from "./markdown-renderer";
import { MessageVersionControls } from "./message-version-controls";
import { ModelSelector } from "./model-selector";
import { PerformanceDropdown } from "./performance-dropdown";

export type MessageType = {
  text: string;
  id: string;
  sender: "assistant" | "user";
  modelId?: string;
  modelName?: string;
  versions?: MessageVersion[];
  currentVersionIndex?: number;
  attachmentIds?: string[];
  error?: string;
};

type MessageProps = MessageType & {
  isLoading?: boolean;
  chatId: string;
  isLast?: boolean;
  isUiStreaming?: boolean;
  isCanceled?: boolean;
  onModelSelect?: (messageId: string, modelId: string, modelName: string) => void;
};

const UserMessage: React.FC<Pick<MessageProps, "text" | "attachmentIds">> = ({ text, attachmentIds }) => (
  <>
    {attachmentIds && attachmentIds.length > 0 && (
      <div className="mb-3 ml-auto w-fit">
        <AttachedFilesDisplay files={attachmentStorage.getFiles(attachmentIds)} />
      </div>
    )}
    <div className="font-[350] text-label-title relative w-fit rounded-[5px] text-[15px] leading-[140%] px-2.5 py-[6px] bg-bg-hover ml-auto [&>div>*:first-child]:mt-0 [&>div>*:last-child]:mb-0">
      <MarkdownRenderer content={text} />
    </div>
  </>
);

const AssistantMessage: React.FC<MessageProps> = ({
  text,
  id,
  versions,
  currentVersionIndex = 0,
  onModelSelect,
  chatId,
  isLoading,
  isLast,
  isUiStreaming,
  isCanceled = false,
}) => {
  const getEffectiveModelForMessage = useChatStore((s) => s.getEffectiveModelForMessage);
  const switchMessageVersion = useChatStore((s) => s.switchMessageVersion);
  const messageFromStore = useChatStore((s) => s.messages.find((m) => m.id === id));

  const { modelId: effectiveModelId, modelName: effectiveModelName } = getEffectiveModelForMessage(id, chatId);

  const storeVersionIndex = messageFromStore?.currentVersionIndex ?? 0;
  const currentMsgVersion = messageFromStore?.versions?.[storeVersionIndex];

  const hasActiveVersions = Array.isArray(messageFromStore?.versions) && (messageFromStore?.versions?.length || 0) > 0;
  const parsed = hasActiveVersions ? currentMsgVersion?.output?.text?.parsed : messageFromStore?.output?.text?.parsed;

  const visibleChainOfThought = parsed?.chainOfThought;
  const visibleMessageText = hasActiveVersions ? (currentMsgVersion?.text ?? text) : text;
  const visibleResponseText = parsed?.response ?? visibleMessageText ?? "";
  const hasError = hasActiveVersions ? !!currentMsgVersion?.error : !!messageFromStore?.error;
  const errorText = hasActiveVersions ? currentMsgVersion?.error : messageFromStore?.error;

  const streamInProgress = !!(isLast && (isLoading || isUiStreaming));

  const [isReasoningVisible, setIsReasoningVisible] = useState(false);
  const autoOpenedByStreamRef = useRef(false);
  const userToggledRef = useRef(false);

  const messageRef = useRef<HTMLDivElement>(null);
  const reasoningScrollRef = useRef<HTMLDivElement>(null);

  const reasoningPinnedRef = useRef(true);

  const reasoningAutoScrollingRef = useRef(false);

  const onReasoningScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (reasoningAutoScrollingRef.current) return;
    const el = e.currentTarget;
    reasoningPinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  useEffect(() => {
    setIsReasoningVisible(false);
    autoOpenedByStreamRef.current = false;
    userToggledRef.current = false;
    reasoningPinnedRef.current = true;
  }, [currentMsgVersion?.id]);

  const stickReasoningToBottom = () => {
    const el = reasoningScrollRef.current;
    if (!el || !reasoningPinnedRef.current) return;
    reasoningAutoScrollingRef.current = true;
    el.scrollTop = el.scrollHeight;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        reasoningAutoScrollingRef.current = false;
      });
    });
  };

  useEffect(() => {
    if (isReasoningVisible) stickReasoningToBottom();
  }, [visibleChainOfThought, isReasoningVisible]);

  useEffect(() => {
    const el = reasoningScrollRef.current;
    if (!isReasoningVisible || !el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => stickReasoningToBottom());
    if (el.firstElementChild) observer.observe(el.firstElementChild);
    return () => observer.disconnect();
  }, [isReasoningVisible]);

  useEffect(() => {
    const shouldAutoOpen =
      !!visibleChainOfThought &&
      !visibleResponseText &&
      !!isUiStreaming &&
      !isReasoningVisible &&
      !userToggledRef.current;
    if (shouldAutoOpen) {
      setIsReasoningVisible(true);
      autoOpenedByStreamRef.current = true;
    }
  }, [visibleChainOfThought, visibleResponseText, isUiStreaming, isReasoningVisible]);

  const handleMessageCopy = async () => {
    if (!messageRef.current) {
      throw new Error("Message element not found");
    }

    const messageClone = messageRef.current.cloneNode(true) as HTMLDivElement;

    const buttons = messageClone.querySelectorAll('button, .copy-button, [class*="copy"]');
    buttons.forEach((button) => button.remove());

    await writeItemsWithFocus([
      new ClipboardItem({
        "text/html": new Blob([messageClone.innerHTML], { type: "text/html" }),
        "text/plain": new Blob([messageClone.textContent || ""], {
          type: "text/plain",
        }),
      }),
    ]);
  };

  const handleModelSelect = (selectedModelId: string, selectedModelName: string) => {
    onModelSelect?.(id, selectedModelId, selectedModelName);
  };

  const handleVersionChange = (versionIndex: number) => {
    switchMessageVersion(id, versionIndex);
  };

  const anyStreamInProgress = !!(isLoading || isUiStreaming);
  const isStreamFinished = !(isLast && isUiStreaming);
  const modelMenuContent = (
    <>
      <ModelMenuIcon />
      <span className="text-[13px]">Model</span>
    </>
  );

  const totalVersions = versions ? versions.length : 1;
  const hasVersions = totalVersions > 1;
  const effectivePerf = hasVersions ? versions?.[currentVersionIndex || 0]?.perf : messageFromStore?.perf;

  const shouldShowPerf = !!(
    effectivePerf &&
    ((effectivePerf.ttftSec ?? 0) > 0 ||
      (effectivePerf.totalSec ?? 0) > 0 ||
      (effectivePerf.tps ?? 0) > 0 ||
      (effectivePerf.tokensOut ?? 0) > 0)
  );

  const isEjecting = useChatSessionStore((s) => s.isEjecting);
  const blockedByOtherChat = useChatSessionStore((s) => isOtherChatGenerating(s, chatId));

  const currentVersionModelName = hasVersions
    ? versions?.[currentVersionIndex || 0]?.modelName
    : messageFromStore?.modelName;

  return (
    <div className="pb-3 text-label-title relative w-full rounded-[5px] text-[15px] leading-[140%]">
      {visibleChainOfThought && (
        <button
          onClick={() => {
            userToggledRef.current = true;
            setIsReasoningVisible((v) => !v);
          }}
          className={twMerge(
            "group mb-1 px-4 pt-4 border border-cell-border w-full rounded-[8px] transition-colors duration-150",
            "hover:[background-color:rgba(0,0,0,0.01)] dark:hover:[background-color:rgba(255,255,255,0.01)]",
          )}
        >
          <div className="flex pb-4 items-center justify-between p-0 group hover:bg-transparent text-label-title">
            <span className="flex items-center gap-2">
              <ThinkingBubbleIcon className="w-[14px] h-[14px]" />

              <span className="text-[13px] font-[400] leading-[130%]">thinking...</span>
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
            enter="transition-all duration-200 ease-out"
            enterFrom="opacity-0 -translate-y-1"
            enterTo="opacity-100 translate-y-0"
            leave="transition-all duration-150 ease-in"
            leaveFrom="opacity-100 translate-y-0"
            leaveTo="opacity-0 -translate-y-1"
          >
            <div className="relative contain-layout">
              <div
                ref={reasoningScrollRef}
                onScroll={onReasoningScroll}
                className="pb-4 text-left max-h-[180px] overflow-y-auto thin-scrollbar"
              >
                <MarkdownRenderer
                  content={visibleChainOfThought || ""}
                  useOneFontSize={true}
                  className="text-[12px] leading-[150%] font-mono text-label-muted [&_p]:mb-1 [&_p]:mt-1"
                />
              </div>
              <div className="group-hover:pointer-events-none opacity-100 group-hover:opacity-0 transition-opacity duration-150 absolute bottom-0 left-0 right-0 h-[96px] z-10 [background:linear-gradient(180deg,rgba(255,255,255,0)_0%,#FFFFFF_100%)] dark:[background:linear-gradient(180deg,rgba(10,10,10,0)_0%,#0A0A0A_100%)]" />
            </div>
          </Transition>
        </button>
      )}
      <div ref={messageRef}>
        {hasError && (
          <div className="p-3 rounded-md border border-error/30 bg-error/10">
            <span className="text-[13px] leading-[150%] text-error">{errorText}</span>
          </div>
        )}
        <MarkdownRenderer content={visibleResponseText} streaming={streamInProgress} />
      </div>
      {(hasVersions ||
        (!streamInProgress && (isStreamFinished || isCanceled || text.length === 0 || !!visibleChainOfThought))) && (
        <div className="flex justify-between items-center mt-6 pr-6">
          <div className="flex items-center gap-1">
            {hasVersions && (
              <MessageVersionControls
                currentVersion={currentVersionIndex}
                totalVersions={totalVersions}
                onVersionChange={handleVersionChange}
                disabledPrevious={anyStreamInProgress}
                currentModelName={currentVersionModelName || effectiveModelName || ""}
              />
            )}
            {(visibleResponseText.length !== 0 || hasError) && (
              <CopyButton className="!min-w-6 !min-h-6" onCopy={handleMessageCopy} />
            )}
          </div>
          <div className="flex items-center gap-4">
            <ModelSelector
              selectedModel={effectiveModelId || ""}
              onModelSelect={handleModelSelect}
              menuContent={modelMenuContent}
              disabled={(anyStreamInProgress && hasVersions) || isEjecting || blockedByOtherChat}
            />
            {shouldShowPerf && (
              <PerformanceDropdown perf={effectivePerf} disabled={anyStreamInProgress && hasVersions} />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const MessageComponent: React.FC<MessageProps> = (props) =>
  props.sender === "user" ? (
    <UserMessage text={props.text} attachmentIds={props.attachmentIds} />
  ) : (
    <AssistantMessage {...props} />
  );

export const Message = React.memo(MessageComponent);
