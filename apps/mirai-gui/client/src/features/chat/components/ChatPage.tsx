import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearch } from "@tanstack/react-router";

import { ChatHeader } from "@/components/chat/ChatHeader";
import { UNTITLED_CHAT_TITLE } from "@/types/chat";
import { useToast } from "@/ui-kit";
import { useRuntimeSessionStore } from "@/stores/useRuntimeSessionStore";
import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import { useGlobalInstructionsStore } from "@/stores/useGlobalInstructionsStore";
import { useSidebarStore } from "@/stores/useSidebarStore";
import { isChatGenerating } from "@/features/runtime/runtimeBusy";
import { useChatModelSelector } from "../hooks/useChatModelSelector";
import { useLlmStream } from "../hooks/useLlmStream";

import { ModelParamsDrawer } from "@/components/ui/model-params-drawer";
import { isCustomParams, useModelParamsStore } from "@/stores/useModelParamsStore";
import { ChatComposer } from "./ChatComposer";
import { ChatMessageList } from "./ChatMessageList";
import { useActiveAssistantBuffer } from "../hooks/useActiveAssistantBuffer";
import { useChatComposerState } from "../hooks/useChatComposerState";
import { useChatRunStatus } from "../hooks/useChatRunStatus";
import { useChatSwitchEffect } from "../hooks/useChatSwitchEffect";
import { useEjectDrivenModelReset } from "../hooks/useEjectDrivenModelReset";
import { useModelSwitch } from "../hooks/useModelSwitch";
import { useRegenerateMessage } from "../hooks/useRegenerateMessage";
import { useSendChatMessage } from "../hooks/useSendChatMessage";
import { useStopChatRun } from "../hooks/useStopChatRun";

type ChatSearchParams = {
  isNew?: boolean;
  model?: string;
  modelName?: string;
};

export function ChatPage() {
  const { chatId } = useParams({ from: "/chat/$chatId" });
  const search = useSearch({ strict: false }) as ChatSearchParams;

  const toast = useToast();

  const isSidebarOpen = useSidebarStore((s) => s.isOpen);
  const globalInstructions = useGlobalInstructionsStore((s) => s.instructions);
  const loadInstructions = useGlobalInstructionsStore((s) => s.loadInstructions);
  const residentSession = useRuntimeSessionStore((s) => s.residentSession);
  const isModelLoading = useChatSessionStore((s) => s.isModelLoading);
  const isTitleGenerating = useChatSessionStore((s) => s.isTitleGenerating);
  const titleGenChatId = useChatSessionStore((s) => s.titleGenChatId);
  const isTitleGeneratingForChat = isTitleGenerating && titleGenChatId === chatId;
  const savedChats = useChatStore((s) => s.savedChats);
  const saveFailureCount = useChatStore((s) => s.saveFailureCount);
  const isChatStreaming = useChatSessionStore((s) => isChatGenerating(s, chatId));
  const loadingMessageId = useChatSessionStore((s) =>
    s.loadingMessage?.chatId === chatId ? s.loadingMessage.messageId : null,
  );
  const canceledMessageId = useChatSessionStore((s) =>
    s.canceledMessage?.chatId === chatId ? s.canceledMessage.messageId : null,
  );

  // The counter outlives this page, so only a rise seen while mounted is news.
  const seenSaveFailuresRef = useRef(saveFailureCount);
  useEffect(() => {
    if (saveFailureCount === seenSaveFailuresRef.current) return;
    seenSaveFailuresRef.current = saveFailureCount;
    toast.error("Failed to save the message. It may be missing after a restart.", { id: "chat-save-failed" });
  }, [saveFailureCount, toast]);

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

  const currentChatTitle = savedChats.find((c) => c.id === chatId)?.title || UNTITLED_CHAT_TITLE;

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
  const globalReasoningEnabled = useModelParamsStore((s) => s.globalReasoningEnabled);
  const modelParamsModified = isCustomParams(selectedModelParams, globalReasoningEnabled);

  return (
    <div className="flex flex-col h-[calc(100vh-24px)] bg-bg dark:bg-bg-dark pt-4 pb-5 px-5">
      <div className="w-full flex flex-col h-full">
        <ChatHeader
          title={currentChatTitle}
          isSidebarOpen={isSidebarOpen}
          isTitleGenerating={isTitleGeneratingForChat}
        />

        <div className="flex-1 overflow-hidden min-h-0 w-full lg:max-w-[800px] mx-auto">
          <ChatMessageList
            isNewChat={!!search.isNew}
            isChatStreaming={isChatStreaming}
            isLoadingStream={isLoading}
            isTitleGeneratingForChat={isTitleGeneratingForChat}
            isModelLoading={isModelLoading}
            hasResidentModel={Boolean(residentSession)}
            scrollTargetId={scrollTargetId}
            onScrolled={() => setScrollTargetId(null)}
            loadingMessageId={loadingMessageId}
            canceledMessageId={canceledMessageId}
            onMessageModelSelect={handleMessageModelSelect}
          />
        </div>

        <div className="flex-shrink-0 w-full lg:max-w-[800px] mx-auto">
          <ChatComposer
            chatId={chatId}
            isChatStreaming={isChatStreaming}
            isLoadingStream={isLoading}
            isTitleGeneratingForChat={isTitleGeneratingForChat}
            autoSelectSuppressed={autoSelectSuppressed}
            selectedModelId={effectiveSelectedModelId}
            hasInstalledModels={hasInstalledModels}
            attachedFiles={composer.attachedFiles}
            canRun={runStatus.canRun}
            onSendMessage={handleSendMessage}
            onStopResponse={handleStopResponse}
            onModelSelect={handleChatModelSelect}
            onFileSelect={composer.attachFile}
            onRemoveFile={composer.removeFile}
            onOpenModelParams={() => setParamsDrawerOpen(true)}
            modelParamsModified={modelParamsModified}
          />
        </div>
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
