import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import { writeItemsWithFocus } from "@/utils/clipboard";
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
import { TextShimmer } from "@/components/ui/text-shimmer";
import { AssistantCat, PendingStatus, type AssistantPhase } from "./assistant-cat";
import { UzuBirdAnswerSync, useUzuBirdTaps } from "./uzu-bird";

const ChartMessage = lazy(() => import("./chart-message"));

const toolLabels = new Map([
  [
    "show_chart",
    {
      running: "Drawing a chart…",
      done: "Drew a chart",
      failed: "Couldn't draw the chart",
      stopped: "Stopped drawing the chart",
    },
  ],
  [
    "get_current_date_time",
    {
      running: "Checking the date and time…",
      done: "Checked the date and time",
      failed: "Couldn't check the date and time",
      stopped: "Stopped checking the date and time",
    },
  ],
  [
    "set_chat_name",
    {
      running: "Renaming the chat…",
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
    running: `Using ${name}…`,
    done: `Used ${name}`,
    failed: `Couldn't use ${name}`,
    stopped: `Stopped using ${name}`,
  }[state];
}

type TranscriptSegment = { kind: "item"; index: number } | { kind: "steps"; start: number; end: number };

/** Groups the transcript for display: each thinking item starts a "steps" segment that takes in
 *  the tool calls and further thinking after it, up to the next text or chart. Tool calls before
 *  any thinking stay on their own. */
function segmentTranscript(items: TranscriptItem[]): TranscriptSegment[] {
  const segments: TranscriptSegment[] = [];
  for (let i = 0; i < items.length; i += 1) {
    if (items[i]!.type !== "thinking") {
      segments.push({ kind: "item", index: i });
      continue;
    }
    let end = i;
    while (end + 1 < items.length && (items[end + 1]!.type === "thinking" || items[end + 1]!.type === "toolCall")) {
      end += 1;
    }
    segments.push({ kind: "steps", start: i, end });
    i = end;
  }
  return segments;
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
  /** Set while the answer has no content yet: the model is loading or the prompt is being read. */
  pendingPhase?: Extract<AssistantPhase, "loadingModel" | "reading"> | null;
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
  pendingPhase = null,
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

  const lastItem = transcript?.at(-1);
  const phase: AssistantPhase = pendingPhase
    ? pendingPhase
    : hasError
      ? /memory/i.test(errorText ?? "")
        ? "outOfMemory"
        : "error"
      : isCanceled
        ? "stopped"
        : !isThinking
          ? "done"
          : transcript !== undefined
            ? lastItem?.type === "thinking" && !lastItem.completed
              ? "thinking"
              : lastItem?.type === "toolCall" && !lastItem.called
                ? "tool"
                : lastItem?.type === "text"
                  ? "writing"
                  : "reading"
            : reasoningInProgress
              ? "thinking"
              : visibleResponseText
                ? "writing"
                : "reading";

  const birdTaps = useUzuBirdTaps();

  const currentVersionModelName = hasVersions
    ? versions?.[currentVersionIndex || 0]?.modelName
    : messageFromStore?.modelName;

  return (
    <div className="relative flex gap-3 in-data-cat-hang:block">
      <div
        {...birdTaps}
        className="flex h-7 w-7 shrink-0 select-none items-center justify-center in-data-cat-hang:absolute in-data-cat-hang:-left-10 in-data-cat-hang:top-0"
      >
        <AssistantCat phase={phase} live={!!isLast} />
      </div>
      <div className="text-label-title relative min-w-0 flex-1 rounded-[5px] text-[15px] leading-[1.6]">
        {pendingPhase ? <PendingStatus phase={pendingPhase} /> : null}
        {isLast ? <UzuBirdAnswerSync answerReady={phase === "done"} /> : null}
        <div
          ref={messageRef}
          className="[&>:last-child]:mb-0 [&>.markdown-body:last-child>div>:last-child]:mb-0 [&>[data-message-reasoning]:has(+.markdown-body>div>*)]:mb-[7px] [&>.markdown-body:first-child]:pt-[3.5px]"
        >
          {transcript !== undefined ? (
            segmentTranscript(transcript).map((segment) => {
              if (segment.kind === "steps") {
                // A run of thinking and tool calls reads as one step of work: one row, with each
                // tool call in place inside the reasoning.
                const steps = transcript.slice(segment.start, segment.end + 1);
                const last = steps[steps.length - 1]!;
                const stillWorking =
                  last.type === "thinking" ? !last.completed : last.type === "toolCall" && !last.called && !last.failed;
                return (
                  <ThinkingBlock
                    key={`${currentMsgVersion?.id ?? id}:${segment.start}`}
                    text={steps
                      .map((step) =>
                        step.type === "toolCall"
                          ? `_${toolCallLabel(step, isThinking)}_`
                          : step.type === "thinking"
                            ? step.text
                            : "",
                      )
                      .filter(Boolean)
                      .join("\n\n")}
                    toolCalls={steps.filter((step) => step.type === "toolCall").length}
                    reasoningInProgress={isThinking && stillWorking && segment.end === transcript.length - 1}
                  />
                );
              }
              const index = segment.index;
              const item = transcript[index]!;
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
                      className="mt-1 mb-2 flex h-6 items-center text-[13px] text-label-muted animate-uzu-in"
                    >
                      {!item.called && !item.failed && isThinking ? (
                        <TextShimmer duration={1.6} className="[--base-color:var(--color-label-muted)]">
                          {toolCallLabel(item, isThinking)}
                        </TextShimmer>
                      ) : (
                        <span key="settled" className="animate-uzu-swap">
                          {toolCallLabel(item, isThinking)}
                        </span>
                      )}
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
            {shouldShowPerf && (
              <PerformanceDropdown perf={effectivePerf} disabled={anyStreamInProgress && hasVersions} />
            )}
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
