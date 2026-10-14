import { useEffect, useRef } from "react";
import { useChatStore } from "@/stores/use-chat-store";

type UseChatSwitchEffectParams = {
  chatId: string;
  replayActiveBuffer: () => void;
  resetLocalUi: () => void;
};

// `replayActiveBuffer` / `resetLocalUi` are read through refs: including them
// in deps would re-trigger clearChat+loadChat on every stream chunk (they
// rebuild with streaming state).
export const useChatSwitchEffect = ({ chatId, replayActiveBuffer, resetLocalUi }: UseChatSwitchEffectParams): void => {
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
    // The URL can retain isNew=true after the first message is saved. Always
    // hydrate from storage; a genuinely new draft simply has no stored chat.
    let stale = false;
    loadChat(chatId)
      .then(() => {
        if (!stale) replayRef.current();
      })
      .catch((e: unknown) => console.error("[chat] failed to load chat", { chatId }, e));
    return () => {
      stale = true;
    };
  }, [chatId, loadChat, setCurrentChatId]);
};
