import { MessageList } from "./message-list";

type ChatMessageListProps = {
  isNewChat: boolean;
  isChatStreaming: boolean;
  isLoadingStream: boolean;
  isTitleGeneratingForChat: boolean;
  isModelLoading: boolean;
  hasResidentModel: boolean;
  scrollTargetId: string | null;
  onScrolled: () => void;
  loadingMessageId: string | null;
  canceledMessageId: string | null;
  onMessageModelSelect: (messageId: string, modelId: string, modelName: string) => void;
};

export const ChatMessageList = ({
  isNewChat,
  isChatStreaming,
  isLoadingStream,
  isTitleGeneratingForChat,
  isModelLoading,
  hasResidentModel,
  scrollTargetId,
  onScrolled,
  loadingMessageId,
  canceledMessageId,
  onMessageModelSelect,
}: ChatMessageListProps) => (
  <MessageList
    isNew={isNewChat}
    isLoading={(isLoadingStream && isChatStreaming) || isTitleGeneratingForChat}
    scrollToMessageId={scrollTargetId}
    onScrolled={onScrolled}
    isUiStreaming={isChatStreaming}
    onMessageModelSelect={onMessageModelSelect}
    loadingMessageId={loadingMessageId}
    canceledMessageId={canceledMessageId}
    isModelLoading={isModelLoading}
    hasResidentModel={hasResidentModel}
  />
);
