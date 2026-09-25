import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import { useCallback } from "react";

type UseStopChatRunParams = {
  chatId: string;
  cancelStream: () => Promise<void>;
};

export const useStopChatRun = ({ chatId, cancelStream }: UseStopChatRunParams) => {
  const stop = useCallback(async () => {
    const session = useChatSessionStore.getState();
    if (!session.canStop()) {
      if (session.titleGenChatId === chatId) await session.cancelActiveRunForChat(chatId);
      return;
    }
    const stoppedMessageId = session.loadingMessage?.chatId === chatId ? session.loadingMessage.messageId : null;
    try {
      await cancelStream();
      if (stoppedMessageId) {
        session.setCanceledMessage(chatId, stoppedMessageId);
        const canceledMsg = useChatStore.getState().messages.find((m) => m.id === stoppedMessageId);
        if (canceledMsg) {
          void useChatStore.getState().persistMessagePatch(chatId, stoppedMessageId, {
            text: canceledMsg.text,
            versions: canceledMsg.versions,
            error: canceledMsg.error,
            output: canceledMsg.output,
          });
        }
      }
    } catch (e) {
      console.warn("[chat] stop failed", e);
    }
  }, [cancelStream, chatId]);

  return { stop };
};
