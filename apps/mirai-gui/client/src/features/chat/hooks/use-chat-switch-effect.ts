import { useEffect, useRef } from "react";
import { useChatStore } from "@/stores/use-chat-store";

type UseChatSwitchEffectParams = {
  chatId: string;
  isNewChat: boolean;
  replayActiveBuffer: () => void;
  resetLocalUi: () => void;
  forgetNewChat: () => void;
};

// The callbacks and `isNewChat` are read through refs: including them in deps
// would re-trigger clearChat+loadChat on every stream chunk (they rebuild with
// streaming state) and when a saved new chat drops its flag.
export const useChatSwitchEffect = ({
  chatId,
  isNewChat,
  replayActiveBuffer,
  resetLocalUi,
  forgetNewChat,
}: UseChatSwitchEffectParams): void => {
  const setCurrentChatId = useChatStore((s) => s.setCurrentChatId);
  const loadChat = useChatStore((s) => s.loadChat);
  const isSaved = useChatStore((s) => s.savedChats.some((chat) => chat.id === chatId));

  const replayRef = useRef(replayActiveBuffer);
  const resetRef = useRef(resetLocalUi);
  const isNewRef = useRef(isNewChat);
  const forgetRef = useRef(forgetNewChat);
  replayRef.current = replayActiveBuffer;
  resetRef.current = resetLocalUi;
  isNewRef.current = isNewChat;
  forgetRef.current = forgetNewChat;

  useEffect(() => {
    if (!chatId) return;
    resetRef.current();
    setCurrentChatId(chatId);
    useChatStore.getState().clearChat();
    if (isNewRef.current) {
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
  }, [chatId, loadChat, setCurrentChatId]);

  // Once saved, a reload must load the chat instead of opening it empty.
  useEffect(() => {
    if (isNewChat && isSaved) forgetRef.current();
  }, [isNewChat, isSaved]);
};
