import { useCallback, useEffect } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { isChatGenerating } from "@/features/runtime/runtime-busy";
import { patchActiveVersion } from "../services/regenerate-versions";

// Replays the buffered streaming text into the in-memory message so a mid-stream
// chat switch or page remount keeps showing the partial output until finalize arrives.
export const useActiveAssistantBuffer = (chatId: string): (() => void) => {
  const generatingHere = useChatSessionStore((s) => isChatGenerating(s, chatId));
  const activeAssistantMessageId = useChatSessionStore((s) => s.activeAssistantMessageId);
  const activeAssistantMessageText = useChatSessionStore((s) => s.activeAssistantMessageText);
  const activeAssistantMessageOutput = useChatSessionStore((s) => s.activeAssistantMessageOutput);

  const apply = useCallback(() => {
    if (!chatId || !generatingHere || !activeAssistantMessageId) return;
    if (activeAssistantMessageText === null && activeAssistantMessageOutput === null) return;
    const s = useChatStore.getState();
    const msg = s.messages.find((m) => m.id === activeAssistantMessageId);
    if (!msg) return;
    const hasVersions = Array.isArray(msg.versions) && msg.versions.length > 0;
    const patch = {
      ...(activeAssistantMessageText !== null ? { text: activeAssistantMessageText } : {}),
      ...(activeAssistantMessageOutput !== null ? { output: activeAssistantMessageOutput } : {}),
    };
    s.updateMessage(activeAssistantMessageId, {
      ...patch,
      ...(hasVersions ? { versions: patchActiveVersion(msg.versions, patch) } : {}),
    });
  }, [chatId, generatingHere, activeAssistantMessageId, activeAssistantMessageText, activeAssistantMessageOutput]);

  useEffect(() => {
    apply();
  }, [apply]);

  return apply;
};
