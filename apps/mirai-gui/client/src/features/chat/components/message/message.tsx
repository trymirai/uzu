import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import { writeItemsWithFocus } from "@/utils/clipboard";
import { WrenchIcon } from "lucide-react";
import React, { lazy, Suspense, useRef } from "react";
import type { MessageVersion } from "@/types/message";
import type { TranscriptItem } from "@/types/llm-stream";
import { useChatStore } from "@/stores/use-chat-store";
import { CopyButton } from "@/components/ui/copy-button";
import { MarkdownRenderer } from "./markdown-renderer";
import { ThinkingBlock } from "./thinking-block";
import { MessageVersionControls } from "./message-version-controls";
import { ModelSelector } from "./model-selector";
import { PerformanceDropdown } from "./performance-dropdown";
import { UserMessage, type UserMessageProps } from "./user-message";

const ChartMessage = lazy(() => import("./chart-message"));

const toolLabels = new Map([
  [
    "show_chart",
    {
      running: "Drawing a chart...",
      done: "Drew a chart",
      failed: "Couldn't draw the chart",
      stopped: "Stopped drawing the chart",
    },
  ],
  [
    "get_current_date_time",
    {
      running: "Checking the date and time...",
      done: "Checked the date and time",
      failed: "Couldn't check the date and time",
      stopped: "Stopped checking the date and time",
    },
  ],
  [
    "set_chat_name",
    {
      running: "Renaming the chat...",
      done: "Renamed the chat",
      failed: "Couldn't rename the chat",
      stopped: "Stopped renaming the chat",
    },
  ],
]);

function toolCallLabel(tool: Extract<TranscriptItem, { type: "toolCall" }>, isRunning: boolean): string {
  const state = tool.failed ? "failed" : tool.called ? "done" : isRunning ? "running" : "stopped";
  const labels = toolLabels.get(tool.name);
  if (labels) return labels[state];

  const name =
    tool.name
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/[_-]+/g, " ")
      .trim() || "tool";
  return {
    running: `Using ${name}...`,
    done: `Used ${name}`,
    failed: `Couldn't use ${name}`,
    stopped: `Stopped using ${name}`,
  }[state];
}

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
  canEdit?: boolean;
  onEdit?: UserMessageProps["onEdit"];
};

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
  const transcript = hasActiveVersions ? currentMsgVersion?.output?.transcript : messageFromStore?.output?.transcript;

  const visibleChainOfThought = parsed?.chainOfThought;
  const visibleMessageText = hasActiveVersions ? (currentMsgVersion?.text ?? text) : text;
  const visibleResponseText = parsed?.response ?? visibleMessageText ?? "";
  const hasError = hasActiveVersions ? !!currentMsgVersion?.error : !!messageFromStore?.error;
  const errorText = hasActiveVersions ? currentMsgVersion?.error : messageFromStore?.error;
  const isGeneratingThisMessage = useChatSessionStore(
    (s) => s.isGenerating && s.activeGeneratingChatId === chatId && s.activeAssistantMessageId === id,
  );
  const isThinking =
    isGeneratingThisMessage &&
    !isCanceled &&
    !hasError &&
    (!hasActiveVersions || currentMsgVersion === messageFromStore?.versions?.at(-1));

  const streamInProgress = !!(isLast && (isLoading || isUiStreaming));
  const reasoningInProgress = isThinking && !visibleResponseText;

  const messageRef = useRef<HTMLDivElement>(null);

  const handleMessageCopy = async () => {
    if (!messageRef.current) {
      throw new Error("Message element not found");
    }

    const messageClone = messageRef.current.cloneNode(true) as HTMLDivElement;

    const buttons = messageClone.querySelectorAll('button, .copy-button, [class*="copy"], [data-tool-call]');
    buttons.forEach((button) => button.remove());
    // A cloned canvas has no bitmap. Copy the chart's readable data instead.
    messageClone.querySelectorAll("canvas").forEach((canvas) => {
      const table = canvas.querySelector("table");
      if (table) canvas.replaceWith(table);
    });
    const plainClone = messageClone.cloneNode(true) as HTMLDivElement;
    plainClone.querySelectorAll("table").forEach((table) => {
      const rows = Array.from(table.rows, (row) => Array.from(row.cells, (cell) => cell.textContent).join("\t"));
      table.replaceWith(document.createTextNode(`\n${rows.join("\n")}\n`));
    });

    await writeItemsWithFocus([
      new ClipboardItem({
        "text/html": new Blob([messageClone.innerHTML], { type: "text/html" }),
        "text/plain": new Blob([plainClone.textContent || ""], {
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
    <div className="text-label-title relative w-full rounded-[5px] text-[15px] leading-[140%]">
      <div
        ref={messageRef}
        className="[&>:last-child]:mb-0 [&>.markdown-body:last-child>div>:last-child]:mb-0 [&>[data-message-reasoning]:has(+.markdown-body>div>*)]:mb-[7px]"
      >
        {transcript !== undefined ? (
          transcript.map((item, index) => {
            const key = `${currentMsgVersion?.id ?? id}:${index}`;
            switch (item.type) {
              case "thinking":
                return (
                  <ThinkingBlock
                    key={key}
                    text={item.text}
                    reasoningInProgress={isThinking && !item.completed && index === transcript.length - 1}
                  />
                );
              case "text":
                return (
                  <MarkdownRenderer
                    key={key}
                    content={item.text}
                    streaming={isThinking && index === transcript.length - 1}
                  />
                );
              case "toolCall":
                return (
                  <div
                    key={key}
                    data-tool-call
                    className="mt-1 mb-2 flex items-center gap-2 pl-1 text-[13px] text-label-muted"
                  >
                    <WrenchIcon className="size-3.5 shrink-0" />
                    <span>{toolCallLabel(item, isThinking)}</span>
                  </div>
                );
              case "chart":
                return (
                  <Suspense key={key} fallback={<div className="my-3 h-96 text-label-muted">Loading chart...</div>}>
                    <ChartMessage chart={item.chart} />
                  </Suspense>
                );
            }
          })
        ) : (
          <>
            {visibleChainOfThought && (
              <ThinkingBlock
                key={currentMsgVersion?.id ?? id}
                text={visibleChainOfThought}
                reasoningInProgress={reasoningInProgress}
                isThinking={isThinking}
              />
            )}
            <MarkdownRenderer content={visibleResponseText} streaming={streamInProgress} />
          </>
        )}
        {hasError && (
          <div role="alert" className="mt-4 p-3 rounded-md border border-error/30 bg-error/10">
            <span className="text-[13px] leading-[150%] text-error">{errorText}</span>
          </div>
        )}
      </div>
      {(hasVersions ||
        (!streamInProgress && (isStreamFinished || isCanceled || text.length === 0 || !!visibleChainOfThought))) && (
        <div className="flex flex-wrap items-center gap-1 mt-2">
          {(visibleResponseText.length !== 0 || hasError || transcript?.some((item) => item.type === "chart")) && (
            <CopyButton className="[&_svg]:size-4" onCopy={handleMessageCopy} />
          )}
          <ModelSelector
            selectedModel={effectiveModelId || ""}
            onModelSelect={handleModelSelect}
            disabled={(anyStreamInProgress && hasVersions) || isEjecting || blockedByOtherChat}
          />
          {shouldShowPerf && <PerformanceDropdown perf={effectivePerf} disabled={anyStreamInProgress && hasVersions} />}
          {hasVersions && (
            <MessageVersionControls
              currentVersion={currentVersionIndex}
              totalVersions={totalVersions}
              onVersionChange={handleVersionChange}
              disabledPrevious={anyStreamInProgress}
              currentModelName={currentVersionModelName || effectiveModelName || ""}
            />
          )}
        </div>
      )}
    </div>
  );
};

const MessageComponent: React.FC<MessageProps> = (props) =>
  props.sender === "user" ? (
    <UserMessage
      id={props.id}
      text={props.text}
      attachmentIds={props.attachmentIds}
      canEdit={props.canEdit}
      onEdit={props.onEdit}
    />
  ) : (
    <AssistantMessage {...props} />
  );

export const Message = React.memo(MessageComponent);
