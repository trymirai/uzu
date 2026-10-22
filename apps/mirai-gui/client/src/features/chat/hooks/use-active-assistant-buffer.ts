import { useCallback, useEffect } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { isChatGenerating } from "@/features/runtime/runtime-busy";
import { patchActiveVersion } from "../services/regenerate-versions";

// Replays the buffered streaming text into the in-memory message so a mid-stream
// chat switch or page remount keeps showing the partial output until finalize arrives.
export const useActiveAssistantBuffer = (chatId: string): (() => void) => {
  const apply = useCallback(() => {
    const session = useChatSessionStore.getState();
    const { activeAssistantMessageId, activeAssistantMessageText, activeAssistantMessageOutput } = session;
    if (!chatId || !isChatGenerating(session, chatId) || !activeAssistantMessageId) return;
    if (activeAssistantMessageText === null && activeAssistantMessageOutput === null) return;
    const s = useChatStore.getState();
    const msg = s.messages.find((m) => m.id === activeAssistantMessageId);
    if (!msg) return;
    const hasVersions = Array.isArray(msg.versions) && msg.versions.length > 0;
    const active = msg.versions?.at(-1) ?? msg;
    // The live stream already updates a loaded message. Replay only when a
    // chat load has replaced it with the older copy from storage.
    if (
      (activeAssistantMessageText === null || active.text === activeAssistantMessageText) &&
      (activeAssistantMessageOutput === null || active.output === activeAssistantMessageOutput)
    ) {
      return;
    }
    const patch = {
      ...(activeAssistantMessageText !== null ? { text: activeAssistantMessageText } : {}),
      ...(activeAssistantMessageOutput !== null ? { output: activeAssistantMessageOutput } : {}),
    };
    s.updateMessage(activeAssistantMessageId, {
      ...patch,
      ...(hasVersions ? { versions: patchActiveVersion(msg.versions, patch) } : {}),
    });
  }, [chatId]);

  useEffect(() => {
    apply();
    return useChatSessionStore.subscribe(apply);
  }, [apply]);

  return apply;
};
