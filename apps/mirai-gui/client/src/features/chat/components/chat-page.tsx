import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { useParams, useSearch } from "@tanstack/react-router";

import { useToast } from "@/components/ui/toast/use-toast";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { useModelsStore } from "@/stores/use-models-store";
import { useAppStore } from "@/stores/use-app-store";
import { useGlobalInstructionsStore } from "@/stores/use-global-instructions-store";
import { isChatGenerating } from "@/features/runtime/runtime-busy";
import { useChatModelSelector } from "../hooks/use-chat-model-selector";
import { useLlmStream } from "../hooks/use-llm-stream";

import { ModelParamsDrawer } from "./composer/model-params-drawer";
import { isCustomParams, useModelParamsStore } from "@/stores/use-model-params-store";
import { ChatComposer } from "./composer/chat-composer";
import { ChatMessageList } from "./message/chat-message-list";
import { useActiveAssistantBuffer } from "../hooks/use-active-assistant-buffer";
import { useChatComposerState } from "../hooks/use-chat-composer-state";
import { useChatRunStatus } from "../hooks/use-chat-run-status";
import { useChatSwitchEffect } from "../hooks/use-chat-switch-effect";
import { useEjectDrivenModelReset } from "../hooks/use-eject-driven-model-reset";
import { useModelSwitch } from "../hooks/use-model-switch";
import { useRegenerateMessage } from "../hooks/use-regenerate-message";
import { useSendChatMessage } from "../hooks/use-send-chat-message";
import { useStopChatRun } from "../hooks/use-stop-chat-run";
import { useMediaQuery } from "@/hooks/use-media-query";

type ChatSearchParams = {
  isNew?: boolean;
  model?: string;
  modelName?: string;
};

export function ChatPage() {
  const { chatId } = useParams({ from: "/chat/$chatId" });
  const search = useSearch({ strict: false }) as ChatSearchParams;

  const toast = useToast();

  const chatWidth = useAppStore((s) => s.chatWidth);
  const globalInstructions = useGlobalInstructionsStore((s) => s.instructions);
  const loadInstructions = useGlobalInstructionsStore((s) => s.loadInstructions);
  const isModelLoading = useChatSessionStore((s) => s.isModelLoading);
  const isSessionBusy = useChatSessionStore(
    (s) => s.operationState !== "idle" || s.isGenerating || s.isTitleGenerating || s.isModelLoading || s.isEjecting,
  );
  const isTitleGenerating = useChatSessionStore((s) => s.isTitleGenerating);
  const titleGenChatId = useChatSessionStore((s) => s.titleGenChatId);
  const isTitleGeneratingForChat = isTitleGenerating && titleGenChatId === chatId;
  const isChatStreaming = useChatSessionStore((s) => isChatGenerating(s, chatId));
  const loadingMessageId = useChatSessionStore((s) =>
    s.loadingMessage?.chatId === chatId ? s.loadingMessage.messageId : null,
  );
  const canceledMessageId = useChatSessionStore((s) =>
    s.canceledMessage?.chatId === chatId ? s.canceledMessage.messageId : null,
  );

  const composer = useChatComposerState();
  const { isLoading, startStream, cancel } = useLlmStream(chatId);

  const {
    selectedChatModel,
    hasAvailableModels: hasInstalledModels,
    currentModelId,
    autoSelectSuppressed,
  } = useChatModelSelector({
    chatId,
    searchModel: search.model,
    searchModelName: search.modelName,
  });

  const [pendingSwitch, setPendingSwitch] = useState(false);

  const [scrollTargetId, setScrollTargetId] = useState<string | null>(null);
  const applyActiveBufferToMessage = useActiveAssistantBuffer(chatId);

  const resetLocalUi = useCallback(() => {
    setScrollTargetId(null);
  }, []);

  // A new chat is a place to type: put the cursor in the composer so you can start right away.
  useEffect(() => {
    if (!search.isNew) return;
    document.querySelector<HTMLTextAreaElement>("textarea[data-composer-input]")?.focus({ preventScroll: true });
  }, [chatId, search.isNew]);

  useChatSwitchEffect({
    chatId,
    isNewChat: !!search.isNew,
    replayActiveBuffer: applyActiveBufferToMessage,
    resetLocalUi,
  });

  useEffect(() => {
    void loadInstructions();
  }, [loadInstructions]);

  const runStatus = useChatRunStatus({ chatId, modelId: currentModelId });

  useEjectDrivenModelReset({
    chatId,
    selectedModelId: selectedChatModel.modelId,
    pendingSwitch,
  });

  const { stop: handleStopResponse } = useStopChatRun({ chatId, cancelStream: cancel });

  const handleSendMessage = useSendChatMessage({
    chatId,
    composer,
    globalInstructions,
    runStatusBlockReason: runStatus.blockReason,
    toast,
    setScrollTargetId,
    startStream,
  });

  const handleEditMessage = useCallback(
    (messageId: string, text: string, onSaved: () => void) => handleSendMessage(text, { messageId, onSaved }),
    [handleSendMessage],
  );

  const handleChatModelSelect = useModelSwitch({
    chatId,
    toast,
    setPendingSwitch,
    cancelStream: cancel,
  });

  const handleMessageModelSelect = useRegenerateMessage({
    chatId,
    globalInstructions,
    toast,
    startStream,
  });

  const effectiveSelectedModelId = autoSelectSuppressed ? "" : currentModelId || "";

  const [paramsDrawerOpen, setParamsDrawerOpen] = useState(false);
  const selectedModelParams = useModelParamsStore((s) =>
    effectiveSelectedModelId ? s.paramsByRepoId[effectiveSelectedModelId] : undefined,
  );
  const globalModelChatNamingEnabled = useModelParamsStore((s) => s.globalModelChatNamingEnabled);
  const selectedModelSize = useModelsStore(
    (s) => s.models.find((model) => model.repoId === effectiveSelectedModelId)?.paramSize,
  );
  const modelParamsModified = isCustomParams(selectedModelParams, globalModelChatNamingEnabled, selectedModelSize);

  // A new, empty chat puts the composer in the middle of the page, under the cat. Once the first
  // message goes out the composer settles at the bottom, gliding there rather than jumping.
  const hasMessages = useChatStore((s) => s.messages.length > 0);
  const centered = !!search.isNew && !hasMessages && !isLoading && !isModelLoading;
  const composerRef = useRef<HTMLDivElement>(null);
  const composerTopRef = useRef<number | null>(null);
  // The conversation runs behind the composer to the panel's edge, so the fade and the room at the
  // end of the list follow the composer's height.
  const [composerHeight, setComposerHeight] = useState(0);
  useLayoutEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    const update = () => setComposerHeight(el.offsetHeight);
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  // The glide belongs to one moment only: the first message going out in this same new chat.
  // Arriving at a chat, from wherever, never animates; the composer is simply where it belongs.
  const settledFromRef = useRef<{ chatId: string; centered: boolean } | null>(null);
  useLayoutEffect(() => {
    const el = composerRef.current;
    const before = composerTopRef.current;
    const last = settledFromRef.current;
    settledFromRef.current = { chatId, centered };
    const sentFirstMessage = last?.chatId === chatId && last.centered && !centered;
    if (!el || before === null || reducedMotion || !sentFirstMessage) return;
    const delta = before - el.getBoundingClientRect().top;
    if (Math.abs(delta) < 2) return;
    el.animate([{ transform: `translateY(${delta}px)` }, { transform: "none" }], {
      duration: 520,
      easing: "cubic-bezier(0.32, 0.72, 0, 1)",
    });
  }, [centered, chatId, reducedMotion]);
  useLayoutEffect(() => {
    composerTopRef.current = composerRef.current?.getBoundingClientRect().top ?? null;
  });

  // Sending with no model chosen: the cat is not pleased and glares at the picker, which nudges.
  const [missingModelAt, setMissingModelAt] = useState(0);
  const handleMissingModel = () => {
    setMissingModelAt(Date.now());
    if (reducedMotion) return;
    document
      .querySelector("[data-model-picker]")
      ?.animate(
        [
          { transform: "translateX(0)" },
          { transform: "translateX(-3px)" },
          { transform: "translateX(3px)" },
          { transform: "translateX(-2px)" },
          { transform: "translateX(0)" },
        ],
        { duration: 360, delay: 450, easing: "cubic-bezier(0.32, 0.72, 0, 1)" },
      );
  };

  return (
    <div
      className="flex flex-col h-full min-h-0 bg-background px-5 pb-5"
      style={{ "--chat-width": `${chatWidth}px`, "--composer-h": `${composerHeight}px` } as CSSProperties}
    >
      <div className="w-full flex flex-col h-full min-h-0">
        {/* A new chat: the cat and its words (about 197px with the air above the ears, plus 32px of
            room above them) sit at the bottom of this area, and the leftover height is shared
            equally above that block and
            below the composer. The space below starts 64px larger, so the group sits 32px above
            the true middle, where the eye expects it. */}

        {/* In a conversation the list reaches behind the composer and the gap under it to the panel's
            edge. A fade covers that stretch: solid page colour in the gap below the composer,
            clearing toward its top edge, so text beside the composer (the cat, wide content) fades
            out as it goes down instead of being cut off. */}
        <div
          className={`${centered ? "flex-[1_1_229px]" : "flex-1"} relative overflow-clip min-h-0 -mx-5`}
          style={centered ? undefined : { marginBottom: "calc(-1 * (var(--composer-h) + 1.25rem))" }}
        >
          <ChatMessageList
            isNewChat={!!search.isNew}
            isChatStreaming={isChatStreaming}
            isLoadingStream={isLoading}
            isTitleGeneratingForChat={isTitleGeneratingForChat}
            isModelLoading={isModelLoading}
            scrollTargetId={scrollTargetId}
            onScrolled={() => setScrollTargetId(null)}
            loadingMessageId={loadingMessageId}
            canceledMessageId={canceledMessageId}
            onMessageModelSelect={handleMessageModelSelect}
            onEditMessage={handleEditMessage}
            canEditMessages={runStatus.canRun && !isSessionBusy}
            draft={composer.draft}
            missingModelAt={missingModelAt}
          />
          {!centered && (
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 bottom-0 h-[calc(var(--composer-h)+1.25rem)] bg-[linear-gradient(to_top,var(--color-background)_1.25rem,transparent)]"
            />
          )}
        </div>

        {/* 8px wider than the text column on each side. */}
        <div
          ref={composerRef}
          className="relative z-10 flex-shrink-0 self-center w-[calc(100%+1rem)] max-w-[calc(var(--chat-width)+1rem)]"
        >
          <ChatComposer
            chatId={chatId}
            isChatStreaming={isChatStreaming}
            isLoadingStream={isLoading}
            isTitleGeneratingForChat={isTitleGeneratingForChat}
            autoSelectSuppressed={autoSelectSuppressed}
            selectedModelId={effectiveSelectedModelId}
            hasInstalledModels={hasInstalledModels}
            draft={composer.draft}
            onDraftChange={composer.setDraft}
            attachedFiles={composer.attachedFiles}
            canRun={runStatus.canRun}
            onSendMessage={handleSendMessage}
            onStopResponse={handleStopResponse}
            onModelSelect={handleChatModelSelect}
            onFileSelect={composer.attachFile}
            onRemoveFile={composer.removeFile}
            onOpenModelParams={() => setParamsDrawerOpen(true)}
            modelParamsModified={modelParamsModified}
            onMissingModel={handleMissingModel}
          />
        </div>
        <div aria-hidden className={centered ? "flex-[1_1_76px] min-h-0" : "flex-[0]"} />
      </div>

      <ModelParamsDrawer
        open={paramsDrawerOpen}
        chatId={chatId}
        repoId={effectiveSelectedModelId || null}
        onClose={() => setParamsDrawerOpen(false)}
      />
    </div>
  );
}
