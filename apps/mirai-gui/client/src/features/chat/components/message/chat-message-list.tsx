import { MessageList } from "./message-list";

type ChatMessageListProps = {
  isNewChat: boolean;
  isChatStreaming: boolean;
  isLoadingStream: boolean;
  isTitleGeneratingForChat: boolean;
  isModelLoading: boolean;
  scrollTargetId: string | null;
  onScrolled: () => void;
  loadingMessageId: string | null;
  canceledMessageId: string | null;
  onMessageModelSelect: (messageId: string, modelId: string, modelName: string) => void;
  onEditMessage: (messageId: string, text: string, onSaved: () => void) => Promise<void>;
  canEditMessages: boolean;
  draft: string;
  missingModelAt: number;
};

export const ChatMessageList = ({
  isNewChat,
  isChatStreaming,
  isLoadingStream,
  isTitleGeneratingForChat,
  isModelLoading,
  scrollTargetId,
  onScrolled,
  loadingMessageId,
  canceledMessageId,
  onMessageModelSelect,
  onEditMessage,
  canEditMessages,
  draft,
  missingModelAt,
}: ChatMessageListProps) => (
  <MessageList
    isNew={isNewChat}
    isLoading={(isLoadingStream && isChatStreaming) || isTitleGeneratingForChat}
    scrollToMessageId={scrollTargetId}
    onScrolled={onScrolled}
    isUiStreaming={isChatStreaming}
    onMessageModelSelect={onMessageModelSelect}
    onEditMessage={onEditMessage}
    canEditMessages={canEditMessages}
    loadingMessageId={loadingMessageId}
    canceledMessageId={canceledMessageId}
    isModelLoading={isModelLoading}
    draft={draft}
    missingModelAt={missingModelAt}
  />
);
