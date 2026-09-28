import { useEffect, useRef } from "react";
import { useChatStore } from "@/stores/use-chat-store";

type UseChatSwitchEffectParams = {
  chatId: string;
  isNewChat: boolean;
  replayActiveBuffer: () => void;
  resetLocalUi: () => void;
};

// `replayActiveBuffer` / `resetLocalUi` are read through refs: including them
// in deps would re-trigger clearChat+loadChat on every stream chunk (they
// rebuild with streaming state).
export const useChatSwitchEffect = ({
  chatId,
  isNewChat,
  replayActiveBuffer,
  resetLocalUi,
}: UseChatSwitchEffectParams): void => {
  const setCurrentChatId = useChatStore((s) => s.setCurrentChatId);
  const loadChat = useChatStore((s) => s.loadChat);

  const replayRef = useRef(replayActiveBuffer);
  const resetRef = useRef(resetLocalUi);
  replayRef.current = replayActiveBuffer;
  resetRef.current = resetLocalUi;

  useEffect(() => {
    if (!chatId) return;
    resetRef.current();
    setCurrentChatId(chatId);
    useChatStore.getState().clearChat();
    if (isNewChat) {
      replayRef.current();
      return;
    }
    let stale = false;
    loadChat(chatId)
      .then(() => {
        if (!stale) replayRef.current();
      })
      .catch((e: unknown) => console.error("[chat] failed to load chat", { chatId }, e));
    return () => {
      stale = true;
    };
  }, [chatId, isNewChat, loadChat, setCurrentChatId]);
};
