import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";

export const INTERRUPTED_REPLY_ERROR = "Mirai closed before the reply finished.";

/** Saves the reply still generating, as far as it got, and marks it interrupted. */
export const saveInterruptedReply = async (): Promise<void> => {
  const session = useChatSessionStore.getState();
  const reply = session.loadingMessage;
  if (!reply) return;
  const shown = useChatStore.getState().messages.find((message) => message.id === reply.messageId)?.text;
  const buffered = session.activeAssistantMessageId === reply.messageId ? session.activeAssistantMessageText : null;
  await useChatStore
    .getState()
    .persistMessageError(reply.chatId, reply.messageId, shown || buffered || "", INTERRUPTED_REPLY_ERROR);
};
