import { useToast } from "@/components/ui/toast/use-toast";
import { useInstalledPickerModels } from "../../hooks/use-picker-models";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import type { AttachedFile } from "@/types/files";
import { MAX_FILES_PER_MESSAGE, SUPPORTED_FILE_TYPES } from "@/constants/attachments";
import { isValidFileSize, isValidFileType, processFile } from "../../services/attachment-files";
import { ChatInput } from "./chat-input";
import { useState } from "react";

type ChatComposerProps = {
  chatId: string;
  isChatStreaming: boolean;
  isLoadingStream: boolean;
  isTitleGeneratingForChat: boolean;
  autoSelectSuppressed: boolean;
  selectedModelId: string;
  hasInstalledModels: boolean;
  attachedFiles: AttachedFile[];
  canRun: boolean;
  onSendMessage: (message: string) => void;
  onStopResponse: () => void;
  onModelSelect: (modelId: string, modelName: string) => void;
  onFileSelect: (file: AttachedFile) => void;
  onRemoveFile: (fileId: string) => void;
  onOpenModelParams: () => void;
  modelParamsModified: boolean;
};

const ATTACH_ACCEPT = SUPPORTED_FILE_TYPES.text.join(",");

export const ChatComposer = ({
  chatId,
  isChatStreaming,
  isLoadingStream,
  isTitleGeneratingForChat,
  autoSelectSuppressed,
  selectedModelId,
  hasInstalledModels,
  attachedFiles,
  canRun,
  onSendMessage,
  onStopResponse,
  onModelSelect,
  onFileSelect,
  onRemoveFile,
  onOpenModelParams,
  modelParamsModified,
}: ChatComposerProps) => {
  const [draft, setDraft] = useState("");
  const toast = useToast();

  const isEjecting = useChatSessionStore((s) => s.isEjecting);
  const otherChatGenerating = useChatSessionStore((s) => isOtherChatGenerating(s, chatId));
  const installedModels = useInstalledPickerModels();

  const blockedMessage =
    !selectedModelId || autoSelectSuppressed
      ? "Select a model to continue"
      : otherChatGenerating
        ? "A response is already generating in another chat."
        : undefined;

  const isLoading = isLoadingStream || isTitleGeneratingForChat;
  const isStreaming = isChatStreaming || isTitleGeneratingForChat;
  const busy = isStreaming || isLoading || isEjecting;

  const handleAttach = async (file: File) => {
    if (attachedFiles.length >= MAX_FILES_PER_MESSAGE) {
      toast.warning(`Maximum ${MAX_FILES_PER_MESSAGE} files allowed per message`);
      return;
    }
    if (!isValidFileType(file)) {
      toast.error("Unsupported file type");
      return;
    }
    if (!isValidFileSize(file, attachedFiles)) {
      toast.error("File too large or exceeds total attachment size limit");
      return;
    }
    try {
      onFileSelect(await processFile(file));
    } catch (error) {
      console.error("Error processing file:", error);
      toast.error("Error processing file");
    }
  };

  return (
    <ChatInput
      value={draft}
      onChange={setDraft}
      placeholder="Add message…"
      onSend={({ text }) => {
        if (text) onSendMessage(text);
      }}
      canSend={({ text }) => !!text.trim() && !isLoading && hasInstalledModels && canRun}
      onBlockedSend={({ text, files }) => {
        if (isLoading || !hasInstalledModels) return;
        if (!text.trim()) {
          if (files.length > 0) toast.error("Add a message to send attachments");
          return;
        }
        toast.error(blockedMessage || "Sending is blocked right now");
      }}
      streaming={isStreaming}
      onStop={onStopResponse}
      onAttach={handleAttach}
      attachAccept={ATTACH_ACCEPT}
      files={attachedFiles.map((f) => ({ id: f.id, name: f.name, extension: f.extension }))}
      onRemoveFile={onRemoveFile}
      models={installedModels}
      activeModelId={selectedModelId}
      onModelChange={(id) => {
        const model = installedModels.find((m) => m.id === id);
        if (model) onModelSelect(id, model.name);
      }}
      modelPickerDisabled={busy || otherChatGenerating}
      moreModelsLink={{ href: "/local-models" }}
      {...(selectedModelId ? { onModelSettingsClick: onOpenModelParams } : {})}
      settingsModified={modelParamsModified}
      settingsDisabled={busy}
    />
  );
};
